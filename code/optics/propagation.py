"""
角谱法（Angular Spectrum Method）自由空间衍射传播。

实现要点：
  - 传递函数 H(fx,fy) = exp(i*2*pi*z*sqrt(1/lambda^2 - fx^2 - fy^2))
  - 倏逝波（sqrt 内为负）置零，避免 NaN 且不让 z<0 时指数爆炸
  - band-limited ASM（Matsushima & Shimobaba 2009）抑制长距离混叠
  - zero-pad 到 2N 再 FFT，裁回 N，避免循环卷积绕回
  - 全程可微（torch.fft），无任何可训练参数（纯物理）

约定：fx,fy 用 torch.fft.fftfreq（未 shift，与 fft2 输出同序），H 同序，故无需 fftshift。
强度一律用 U.real**2 + U.imag**2（不要 torch.abs），避免 0 振幅点梯度 NaN。
"""

import math
import torch
import torch.nn.functional as F


def check_sampling(dx, wavelength, strict=True):
    """强制 SLM 像素间距 >= 2*lambda（提醒 b）。返回是否满足；strict 时不满足抛错。"""
    ratio = dx / wavelength
    ok = ratio >= 2.0 - 1e-9
    if not ok and strict:
        raise ValueError(
            f"像素间距 dx={dx*1e6:.4f} um 必须 >= 2*lambda = {2*wavelength*1e6:.4f} um "
            f"(当前 dx/lambda={ratio:.3f})"
        )
    return ok


def angular_spectrum_propagate(u, z, dx, wavelength, band_limit=True, pad_factor=2):
    """
    band-limited 角谱传播。

    参数
    ----
    u : 复数张量 [..., H, W]（complex64 或 complex128）—— 输入光场
    z : float —— 传播距离（米），可正可负（负=反向传播）
    dx : float —— 像素间距（米）
    wavelength : float —— 波长（米）
    band_limit : bool —— 是否套 Matsushima 2009 带限掩膜抑制混叠
    pad_factor : int —— FFT 前 zero-pad 倍数（默认 2）

    返回
    ----
    传播后的复数张量 [..., H, W]，dtype 与输入一致。
    """
    if u.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("u 必须是复数张量 (complex64/complex128)")
    real_dtype = torch.float64 if u.dtype == torch.complex128 else torch.float32

    H, W = u.shape[-2:]
    ph = (pad_factor - 1) * H // 2
    pw = (pad_factor - 1) * W // 2
    u_pad = F.pad(u, (pw, pw, ph, ph))          # 四周补零
    Hp, Wp = u_pad.shape[-2:]

    # 频率网格（cycles/m），未 shift 以匹配 fft2
    fx = torch.fft.fftfreq(Wp, d=dx, device=u.device, dtype=real_dtype)
    fy = torch.fft.fftfreq(Hp, d=dx, device=u.device, dtype=real_dtype)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    # sqrt 内的量；>0 为传播波，<=0 为倏逝波
    arg = (1.0 / wavelength**2) - FX**2 - FY**2
    propagating = (arg > 0).to(real_dtype)       # 倏逝波置零（forward/backward 对称）
    kz = 2.0 * math.pi * torch.sqrt(arg.clamp(min=0.0))   # 仅传播波有效，余者被掩膜清零

    Htf = torch.exp(1j * (kz * z)) * propagating  # 纯相位传递函数（|H|=1 在传播带）

    if band_limit:
        # Matsushima & Shimobaba 2009 带限：抑制长 z 下传递函数欠采样混叠
        Lx = Wp * dx
        Ly = Hp * dx
        fx_max = 1.0 / (wavelength * math.sqrt((2.0 * abs(z) / Lx) ** 2 + 1.0))
        fy_max = 1.0 / (wavelength * math.sqrt((2.0 * abs(z) / Ly) ** 2 + 1.0))
        bl = ((FX.abs() < fx_max) & (FY.abs() < fy_max)).to(real_dtype)
        Htf = Htf * bl

    Htf = Htf.to(u.dtype)
    out = torch.fft.ifft2(torch.fft.fft2(u_pad) * Htf)
    return out[..., ph:ph + H, pw:pw + W]        # 裁回原尺寸


def intensity(u):
    """平方律探测 I = |U|^2，用 real^2+imag^2 而非 abs，保证 0 点梯度安全。"""
    return u.real ** 2 + u.imag ** 2
