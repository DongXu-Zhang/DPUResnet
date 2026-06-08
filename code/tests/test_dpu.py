"""
Step 2 测试：DPU 模块（独立脚本，无需 pytest）

运行： python code/tests/test_dpu.py
重点：⭐ 线性消融对照 —— 证明 |U|² 探测是非线性的真正来源（印证老师"非线性很重要"）。
覆盖：唯一可训练=φ / φ梯度有限 / 强度非负 / φ=0退化为纯传播 / 形状 / ⭐线性消融对照。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
from optics.dpu import DPUBlock, DPUStack  # noqa: E402
from optics.propagation import angular_spectrum_propagate as asm  # noqa: E402

torch.manual_seed(0)


def test_only_phi_trainable():
    # 合规核心：唯一可训练参数必须是相位掩膜 phi
    stack = DPUStack(size=16, n_blocks=2)
    names = stack.trainable_param_names()
    assert len(names) > 0, "竟然没有可训练参数"
    assert all("phi" in n for n in names), f"出现非φ的可训练参数: {names}"
    return f"可训练参数全部是φ（{len(names)}个）"


def test_phi_grad_finite():
    stack = DPUStack(size=32, n_blocks=3)
    img = torch.rand(2, 32, 32)
    target = torch.rand(2, 32, 32)
    loss = ((stack(img) - target) ** 2).mean()
    loss.backward()
    for n, p in stack.named_parameters():
        assert p.grad is not None, f"{n} 无梯度"
        assert torch.isfinite(p.grad).all(), f"{n} 梯度含 NaN/Inf"
    return "所有φ梯度有限"


def test_intensity_nonneg():
    stack = DPUStack(size=32, n_blocks=3)
    out = stack(torch.rand(1, 32, 32))
    assert (out >= 0).all(), "强度输出出现负值"
    assert not out.is_complex(), "输出应为实数强度"
    return f"min={out.min().item():.2e} >=0"


def test_phi_zero_pure_propagation():
    # φ=0 时 DPUBlock 应退化为纯角谱传播
    blk = DPUBlock(size=32, n_phase=1).double()
    blk.phi.data.zero_()
    U = torch.randn(32, 32, dtype=torch.complex128)
    out = blk(U)
    ref = asm(U, blk.z, blk.dx, blk.wavelength)
    err = (out - ref).abs().max().item()
    assert err < 1e-9, f"φ=0 未退化为纯传播, 误差={err:.2e}"
    return f"max误差={err:.1e}"


def test_forward_shape():
    stack = DPUStack(size=64, n_blocks=2)
    img = torch.rand(3, 64, 64)
    out = stack(img)
    assert out.shape == img.shape, f"形状不符: {out.shape} vs {img.shape}"
    return f"{tuple(img.shape)} -> {tuple(out.shape)}"


def test_linearity_collapse_control():
    # ⭐ 关键对照：关掉 block 间 |U|² 探测 -> 场级映射严格线性(叠加性成立)
    #            开启 |U|² 探测 -> 叠加性被破坏 -> 非线性确实来自平方探测
    stack = DPUStack(size=32, n_blocks=3).double()   # 双精度求严格叠加
    U1 = torch.randn(32, 32, dtype=torch.complex128)
    U2 = torch.randn(32, 32, dtype=torch.complex128)
    a = torch.tensor(0.7 + 0.3j, dtype=torch.complex128)
    b = torch.tensor(-0.4 + 0.9j, dtype=torch.complex128)

    # 线性模式：f(aU1+bU2) 应 == a f(U1)+b f(U2)
    f0 = lambda U: stack.process_field(U, detect_between=False)
    lhs = f0(a * U1 + b * U2)
    rhs = a * f0(U1) + b * f0(U2)
    lin_err = ((lhs - rhs).abs().max() / rhs.abs().max()).item()

    # 非线性模式：叠加性应被明显破坏
    f1 = lambda U: stack.process_field(U, detect_between=True)
    lhs2 = f1(a * U1 + b * U2)
    rhs2 = a * f1(U1) + b * f1(U2)
    nl_err = ((lhs2 - rhs2).abs().max() / rhs2.abs().max()).item()

    assert lin_err < 1e-6, f"关探测后应严格线性, 但叠加误差={lin_err:.2e}"
    assert nl_err > 1e-2, f"开|U|²探测后应非线性, 但叠加误差仅={nl_err:.2e}"
    return f"线性模式叠加误差={lin_err:.1e}(线性) | 非线性模式={nl_err:.1e}(破坏)"


TESTS = [
    test_only_phi_trainable,
    test_phi_grad_finite,
    test_intensity_nonneg,
    test_phi_zero_pure_propagation,
    test_forward_shape,
    test_linearity_collapse_control,
]

if __name__ == "__main__":
    n_pass = 0
    for t in TESTS:
        try:
            detail = t()
            print(f"  ✅ {t.__name__:<34} {detail}")
            n_pass += 1
        except Exception as e:
            print(f"  ❌ {t.__name__:<34} {type(e).__name__}: {e}")
    print(f"\nStep2(DPU模块): {n_pass}/{len(TESTS)} 通过")
    sys.exit(0 if n_pass == len(TESTS) else 1)
