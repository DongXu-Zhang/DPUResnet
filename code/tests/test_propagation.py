"""
安全门 1：角谱传播器单元测试（独立脚本，无需 pytest）

运行： python code/tests/test_propagation.py
全部通过才算地基可信，才能往 DPU 模块走。
覆盖：z=0 恒等 / 正反传播往返 / 能量守恒 / 高斯光束扩散对解析解 /
      gradcheck 无 NaN / 0 振幅点梯度安全 / 倏逝波不爆 / pitch>=2*lambda 校验
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
from optics.propagation import (  # noqa: E402
    angular_spectrum_propagate as asm,
    intensity,
    check_sampling,
)
from configs.config import WAVELENGTH as LAM, PIXEL_PITCH as DX, PATCH as N  # noqa: E402

torch.manual_seed(0)


def _rms_radius(I):
    """二维强度的二阶矩半径（像素）。"""
    H, W = I.shape
    ys = torch.arange(H, dtype=I.dtype)
    xs = torch.arange(W, dtype=I.dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    tot = I.sum()
    cy = (I * Y).sum() / tot
    cx = (I * X).sum() / tot
    var = (I * ((X - cx) ** 2 + (Y - cy) ** 2)).sum() / tot
    return torch.sqrt(var).item()


# ---------------------------------------------------------------- 测试 --------
def test_z0_identity():
    u = torch.randn(N, N, dtype=torch.complex128)
    out = asm(u, 0.0, DX, LAM)
    err = (out - u).abs().max().item()
    assert err < 1e-9, f"z=0 应为恒等, 最大误差={err:.2e}"
    return f"max|out-u|={err:.1e}"


def test_roundtrip():
    # 平滑高斯场，正向 z 再反向 -z 应还原
    z = 50e-6
    yy, xx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    r2 = ((xx - N / 2) ** 2 + (yy - N / 2) ** 2).to(torch.float64)
    u = torch.exp(-r2 / (2 * (12.0 ** 2))).to(torch.complex128)
    back = asm(asm(u, z, DX, LAM), -z, DX, LAM)
    rel = (back - u).abs().pow(2).sum().sqrt() / u.abs().pow(2).sum().sqrt()
    rel = rel.item()
    assert rel < 1e-3, f"正反传播往返相对误差过大: {rel:.2e}"
    return f"relL2={rel:.1e}"


def test_energy_conservation():
    z = 50e-6
    yy, xx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    r2 = ((xx - N / 2) ** 2 + (yy - N / 2) ** 2).to(torch.float64)
    u = torch.exp(-r2 / (2 * (12.0 ** 2))).to(torch.complex128)
    e_in = intensity(u).sum().item()
    e_out = intensity(asm(u, z, DX, LAM)).sum().item()
    rel = abs(e_out - e_in) / e_in
    assert rel < 0.01, f"能量不守恒, 相对变化={rel:.3%}"
    return f"dE={rel:.2%}"


def test_gaussian_spreading():
    # 高斯光束应扩散，且与近轴解析解 w(z)=w0*sqrt(1+(z/zR)^2) 大致吻合
    w0_px = 10.0
    w0 = w0_px * DX
    zR = math.pi * w0 ** 2 / LAM          # 瑞利距离
    z = zR                                # 取一个瑞利距离，理论展宽 sqrt(2)
    yy, xx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    r2 = ((xx - N / 2) ** 2 + (yy - N / 2) ** 2).to(torch.float64)
    u = torch.exp(-r2 / (w0_px ** 2)).to(torch.complex128)   # 振幅高斯
    w_before = _rms_radius(intensity(u))
    w_after = _rms_radius(intensity(asm(u, z, DX, LAM)))
    ratio_meas = w_after / w_before
    ratio_theo = math.sqrt(1 + (z / zR) ** 2)               # = sqrt(2)
    assert w_after > w_before, "高斯光束应扩散但没有"
    rel = abs(ratio_meas - ratio_theo) / ratio_theo
    assert rel < 0.25, f"展宽比偏离近轴解析解 {rel:.1%} (实测{ratio_meas:.3f} vs 理论{ratio_theo:.3f})"
    return f"展宽比 实测{ratio_meas:.3f}/理论{ratio_theo:.3f}"


def test_gradcheck():
    z = 20e-6
    n = 6
    phi = torch.rand(n, n, dtype=torch.float64, requires_grad=True)

    def f(p):
        u = torch.exp(1j * p.to(torch.complex128))   # 振幅恒为1，无0点
        return intensity(asm(u, z, DX, LAM))

    ok = torch.autograd.gradcheck(f, (phi,), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert ok, "gradcheck 未通过"
    return "gradcheck OK"


def test_grad_no_nan_with_zeros():
    # 含精确 0 振幅像素时，前向与梯度都不能出现 NaN
    z = 30e-6
    amp = torch.ones(N, N, dtype=torch.float64)
    amp[::3, ::3] = 0.0                               # 撒一堆 0
    phi = torch.zeros(N, N, dtype=torch.float64, requires_grad=True)
    u = amp * torch.exp(1j * phi.to(torch.complex128))
    out = asm(u, z, DX, LAM)
    assert torch.isfinite(out).all(), "前向出现 NaN/Inf"
    intensity(out).sum().backward()
    assert torch.isfinite(phi.grad).all(), "梯度出现 NaN/Inf"
    return "前向/梯度均有限"


def test_evanescent_no_blowup():
    # 反向传播（z<0）时倏逝波若不置零会指数爆炸；这里应保持有界
    z = -200e-6
    u = torch.randn(N, N, dtype=torch.complex128)
    out = asm(u, z, DX, LAM)
    assert torch.isfinite(out).all(), "反向传播出现 NaN/Inf（倏逝波处理有问题）"
    return f"max|out|={out.abs().max().item():.2e} 有界"


def test_pitch_constraint():
    assert check_sampling(2 * LAM, LAM) is True
    assert check_sampling(1.5 * LAM, LAM, strict=False) is False
    raised = False
    try:
        check_sampling(1.5 * LAM, LAM, strict=True)
    except ValueError:
        raised = True
    assert raised, "strict 模式下 pitch<2*lambda 应抛错"
    # 当前配置必须满足
    assert check_sampling(DX, LAM)
    return f"dx/lambda={DX/LAM:.2f}"


# ---------------------------------------------------------------- runner ------
TESTS = [
    test_z0_identity,
    test_roundtrip,
    test_energy_conservation,
    test_gaussian_spreading,
    test_gradcheck,
    test_grad_no_nan_with_zeros,
    test_evanescent_no_blowup,
    test_pitch_constraint,
]

if __name__ == "__main__":
    print(f"配置: lambda={LAM*1e9:.0f}nm  dx={DX*1e6:.3f}um (={DX/LAM:.1f}*lambda)  patch={N}")
    n_pass = 0
    for t in TESTS:
        try:
            detail = t()
            print(f"  ✅ {t.__name__:<28} {detail}")
            n_pass += 1
        except Exception as e:
            print(f"  ❌ {t.__name__:<28} {type(e).__name__}: {e}")
    print(f"\n安全门1: {n_pass}/{len(TESTS)} 通过")
    sys.exit(0 if n_pass == len(TESTS) else 1)
