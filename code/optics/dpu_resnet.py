"""
DPU-ResNet —— 文档要求的"基于 DPU 的 ResNet"（核心交付物）。

残差全光实现（Res-D2NN, Dou 2020）：每个残差块内，残差支路 F(U) 与捷径 U 在
**探测之前做相干复场相加**（固定 50/50 分束器，无可训练权重），再过 |U|² 探测(非线性)、
sqrt 重编码。跨块走强度重编码。唯一可训练参数仍只有相位掩膜 φ。

  ResidualDPUBlock:  U -> F=相位+衍射(U);  merged=(F+U)/√2(相干跳连);  I=|merged|²;  返回 sqrt(I)
  DPUResNet:         编码 -> [ResidualDPUBlock]×N -> 末端读 |U|²

说明：相位-only 无损，不采用"全局残差 out=WF+correction"(与能量守恒冲突)；
     输出最终强度由训练脚本做固定归一化(/max)对齐 [0,1]。
"""

import math
import torch
import torch.nn as nn

from optics.propagation import angular_spectrum_propagate as asm, intensity
from optics.dpu import _real_dtype, _R2C
from configs.config import (
    WAVELENGTH, PIXEL_PITCH, EPS, Z_DEFAULT, N_BLOCKS, N_PHASE_PER_BLOCK,
)


class ResidualDPUBlock(nn.Module):
    """相干残差块：(F(U)+U)/√2 -> |·|² -> sqrt 重编码。唯一可训练=φ。"""

    def __init__(self, size, n_phase=N_PHASE_PER_BLOCK, z=Z_DEFAULT,
                 dx=PIXEL_PITCH, wavelength=WAVELENGTH, shortcut="sharp", detect=True):
        super().__init__()
        self.phi = nn.Parameter(0.01 * torch.randn(n_phase, size, size))
        self.z, self.dx, self.wavelength = z, dx, wavelength
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)        # 固定 50/50 分束器，非可训练
        self.shortcut = shortcut                     # "sharp"(4f中继锚点+剥piston) | "prop"(自由传播,模糊)
        self.detect = detect                         # False=|U|²消融(块间不探测,保复场→整网线性)
        # 单跳 DC 载波(piston)相位 exp(i·2πz/λ)；sharp 模式用其共轭剥离, 与无piston的锐利捷径匹配
        piston = 2.0 * math.pi * z / wavelength
        self._pcorr = complex(math.cos(piston), -math.sin(piston))   # exp(-i·piston)

    def forward(self, U):
        e_in = intensity(U).sum(dim=(-1, -2), keepdim=True)   # 块输入能量
        F = U
        S = U                                        # 捷径分支(Res-D2NN跳连)
        for l in range(self.phi.shape[0]):
            phase = self.phi[l].to(_real_dtype(U))
            F = F * torch.exp(1j * phase)            # 相位调制(论文2)
            F = asm(F, self.z, self.dx, self.wavelength)  # 残差支路:相位+衍射
            if self.shortcut == "prop":
                S = asm(S, self.z, self.dx, self.wavelength)  # 捷径同距自由传播(模糊→丢锚点)
            else:                                    # "sharp": 4f中继保锐利捷径, 剥F的piston与之匹配
                F = F * self._pcorr
        # ⭐两臂须共享载波相位参考, 否则 cos(2πz/λ)≈−1(如z≈45µm) 两臂DC相消→e_m暴跌→
        #   能量归一化√(e_in/e_m)反将高频残差逐块放大~2.5^N→谐振崩溃(已诊断证实)。
        #   sharp: 锐利捷径(4f中继可实现)做空间锚点, F剥piston匹配 → 既消谐振又保留高分辨锚点(性能好)。
        #   prop:  捷径也传播, 两臂自然共piston → 消谐振但锚点被模糊(性能差)。
        merged = (F + S) * self.inv_sqrt2            # 相干跳连(探测前相加)
        if not self.detect:                          # |U|²消融:严格线性场传播,不探测不归一化
            return merged                            #   整网=单个线性算子,仅末端一次|U|²读出
        e_m = intensity(merged).sum(dim=(-1, -2), keepdim=True)
        merged = merged * torch.sqrt(e_in / (e_m + EPS))  # 固定能量归一化(参数自由全局标量)
        I = intensity(merged)                        # 平方律探测(论文3 非线性)
        return torch.sqrt(I + EPS).to(U.dtype)       # 重编码为下一层振幅


class DPUResNet(nn.Module):
    """基于 DPU 的 ResNet：编码 -> 残差块堆叠 -> 末端读 |U|²。"""

    def __init__(self, size=128, n_blocks=N_BLOCKS, n_phase=N_PHASE_PER_BLOCK,
                 z=Z_DEFAULT, dx=PIXEL_PITCH, wavelength=WAVELENGTH, encode="amplitude",
                 shortcut="sharp", linear=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualDPUBlock(size, n_phase, z, dx, wavelength, shortcut, detect=not linear)
            for _ in range(n_blocks)
        ])
        self.encode = encode

    def encode_field(self, image):
        img = image.clamp(min=0.0)
        if self.encode == "amplitude":
            return torch.sqrt(img + EPS).to(_R2C[img.dtype])
        raise ValueError(f"未知 encode={self.encode}")

    def forward(self, image):                        # image [B,H,W] in [0,1]
        U = self.encode_field(image)
        for blk in self.blocks:
            U = blk(U)
        return intensity(U)                          # 末端 |U|² = 超分辨预测

    def trainable_param_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]
