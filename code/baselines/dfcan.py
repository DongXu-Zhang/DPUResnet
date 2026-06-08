"""
DFCAN 的复现。

仅作"光学 vs 电子"对标用，绝不接入光学网络。结构：
  Conv(head) -> [ResidualGroup × n_group] -> Conv(up) -> Conv(tail) -> sigmoid
  ResidualGroup = [FCAB × n_fcab] + 长跳连
  FCAB = Conv-GELU-Conv-GELU -> FourierChannelAttention -> 加 input
  FourierChannelAttention(核心创新) = 用特征的 FFT 幅度谱(pow0.1压缩)算逐通道注意力权重,
    因为超分辨本质是恢复高频, 频域通道注意力天然契合。

我们 BioSR 子集是同网格 128→128, scale=1(无 pixel-shuffle 放大)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierChannelAttention(nn.Module):
    """逐通道注意力, 权重来自特征 FFT 幅度谱。"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, 3, padding=1)     # 作用在幅度谱上
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                              # 频域全局平均池化
            nn.Conv2d(channel, channel // reduction, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        fft = torch.fft.fft2(x.to(torch.complex64))
        mag = torch.fft.fftshift(torch.abs(fft), dim=(-2, -1))
        mag = torch.pow(mag + 1e-8, 0.1)                          # 动态范围压缩(论文 pow0.1)
        w = F.relu(self.conv(mag.to(x.dtype)))
        w = self.se(w)
        return x * w


class FCAB(nn.Module):
    """Fourier Channel Attention Block: Conv-GELU-Conv-GELU -> FCA -> 残差。"""

    def __init__(self, channel):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1), nn.GELU(),
            nn.Conv2d(channel, channel, 3, padding=1), nn.GELU(),
        )
        self.fca = FourierChannelAttention(channel)

    def forward(self, x):
        return x + self.fca(self.body(x))


class ResidualGroup(nn.Module):
    def __init__(self, channel, n_fcab=4):
        super().__init__()
        self.body = nn.Sequential(*[FCAB(channel) for _ in range(n_fcab)])

    def forward(self, x):
        return x + self.body(x)


class DFCAN(nn.Module):
    def __init__(self, in_ch=1, channel=64, n_group=4, n_fcab=4, scale=1):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_ch, channel, 3, padding=1), nn.GELU())
        self.body = nn.Sequential(*[ResidualGroup(channel, n_fcab) for _ in range(n_group)])
        if scale > 1:
            self.up = nn.Sequential(
                nn.Conv2d(channel, channel * scale * scale, 3, padding=1), nn.GELU(),
                nn.PixelShuffle(scale),
            )
        else:
            self.up = nn.Sequential(nn.Conv2d(channel, channel, 3, padding=1), nn.GELU())
        self.tail = nn.Conv2d(channel, in_ch, 3, padding=1)

    def forward(self, x):                            # x [B,1,H,W] in [0,1]
        f = self.head(x)
        f = f + self.body(f)                         # 全局长跳连
        f = self.up(f)
        return torch.sigmoid(self.tail(f))           # [B,1,H,W] in [0,1]
