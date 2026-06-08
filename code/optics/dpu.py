"""
DPU 模块 —— 结合论文2(Lin/D2NN)与论文3(Zhou/DPU)。

唯一可训练参数 = 相位掩膜 phi（薄元近似，逐像素）。其余全是固定物理：
  编码(图像->复场振幅)、角谱衍射、|U|²平方律探测(非线性)、sqrt重编码(光电环)。

DPUBlock : n_phase 层 [相位 exp(iφ) -> 角谱衍射]，返回复场(不探测)   <- 论文2 相干衍射线性核
DPUStack : 编码 -> 逐 block；block 间 |U|²探测+sqrt重编码 -> 末端读 |U|²  <- 论文3 光电环非线性
           nonlinear=False 时关掉 block 间探测 -> 整栈塌缩成单一线性算子(消融对照)

设计旋钮(可扫)：n_blocks(深度)、n_phase(每block相位层数)、z(层间距)、encode(振幅/相位)。
合规：forward 里只有 phi 是 nn.Parameter；编码/探测/重编码/归一化均无可训练参数。
"""

import torch
import torch.nn as nn

from optics.propagation import angular_spectrum_propagate as asm, intensity
from configs.config import (
    WAVELENGTH, PIXEL_PITCH, EPS, Z_DEFAULT, N_BLOCKS, N_PHASE_PER_BLOCK,
)

_R2C = {torch.float32: torch.complex64, torch.float64: torch.complex128}


def _real_dtype(t):
    return torch.float64 if t.dtype in (torch.complex128, torch.float64) else torch.float32


class DPUBlock(nn.Module):
    """一个 DPU 的相干衍射部分：n_phase 层 (相位调制 -> 衍射)。返回复场，不做探测。"""

    def __init__(self, size, n_phase=N_PHASE_PER_BLOCK, z=Z_DEFAULT,
                 dx=PIXEL_PITCH, wavelength=WAVELENGTH):
        super().__init__()
        # 唯一可训练参数：相位掩膜，初始化在 0 附近(近恒等，训练稳定)
        self.phi = nn.Parameter(0.01 * torch.randn(n_phase, size, size))
        self.z, self.dx, self.wavelength = z, dx, wavelength

    def forward(self, U):
        for l in range(self.phi.shape[0]):
            phase = self.phi[l].to(_real_dtype(U))      # 跟随场精度
            U = U * torch.exp(1j * phase)               # 薄元相位调制(论文2)
            U = asm(U, self.z, self.dx, self.wavelength)  # 角谱衍射
        return U


class DPUStack(nn.Module):
    """堆叠 DPU。block 之间做平方律探测+重编码(非线性)。末端读出强度=超分辨预测。"""

    def __init__(self, size, n_blocks=N_BLOCKS, n_phase=N_PHASE_PER_BLOCK,
                 z=Z_DEFAULT, dx=PIXEL_PITCH, wavelength=WAVELENGTH,
                 nonlinear=True, encode="amplitude"):
        super().__init__()
        self.blocks = nn.ModuleList([
            DPUBlock(size, n_phase, z, dx, wavelength) for _ in range(n_blocks)
        ])
        self.nonlinear = nonlinear
        self.encode = encode

    # --- 固定物理：图像 -> 输入复场（无可训练参数）---
    def encode_field(self, image):
        img = image.clamp(min=0.0)
        cdt = _R2C[img.dtype]
        if self.encode == "amplitude":          # |U0|^2 = image
            return torch.sqrt(img + EPS).to(cdt)
        elif self.encode == "phase":            # 相位编码(可选旋钮)
            phase = (img / (img.amax() + EPS)) * torch.pi
            return torch.exp(1j * phase.to(img.dtype))
        raise ValueError(f"未知 encode={self.encode}")

    # --- 核心：逐 block 处理复场；detect_between 控制 block 间是否插入 |U|² 非线性 ---
    def process_field(self, U, detect_between):
        n = len(self.blocks)
        for i, block in enumerate(self.blocks):
            U = block(U)
            if detect_between and i < n - 1:      # 末块后不探测(留给 forward 读出)
                I = intensity(U)                  # 平方律探测(论文3 非线性)
                U = torch.sqrt(I + EPS).to(U.dtype)  # 重编码为下一层振幅(丢相位,光电环)
        return U

    def forward(self, image):
        U = self.encode_field(image)
        U = self.process_field(U, detect_between=self.nonlinear)
        return intensity(U)                       # 末端探测 |U|^2 = 超分辨预测(实, 非负)

    # --- 合规自检：唯一可训练参数必须是相位掩膜 ---
    def trainable_param_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]
