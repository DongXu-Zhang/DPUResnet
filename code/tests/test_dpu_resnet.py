"""Step 6 测试：DPUResNet（独立脚本）。运行: python code/tests/test_dpu_resnet.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
from optics.dpu_resnet import DPUResNet, ResidualDPUBlock  # noqa: E402

torch.manual_seed(0)


def test_only_phi_trainable():
    net = DPUResNet(size=32, n_blocks=3)
    names = net.trainable_param_names()
    assert len(names) > 0 and all("phi" in n for n in names), f"非φ可训练参数: {names}"
    return f"可训练参数全是φ（{len(names)}个）"


def test_forward_and_grad():
    net = DPUResNet(size=64, n_blocks=4)
    img = torch.rand(2, 64, 64)
    out = net(img)
    assert out.shape == img.shape and not out.is_complex(), "输出形状/类型错误"
    assert (out >= 0).all(), "强度出现负值"
    (((out - torch.rand(2, 64, 64)) ** 2).mean()).backward()
    gfin = all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert gfin, "φ梯度含 NaN/Inf"
    return f"{tuple(img.shape)}->{tuple(out.shape)}, 非负, 梯度有限"


def test_residual_skip_energy_stable():
    # 相干残差块应保持能量大致稳定(不爆不消)，多块堆叠后仍有界
    net = DPUResNet(size=64, n_blocks=8).double()
    img = torch.rand(1, 64, 64, dtype=torch.float64)
    out = net(img)
    assert torch.isfinite(out).all(), "深层堆叠出现 NaN/Inf"
    ratio = out.sum().item() / (img.sum().item() + 1e-9)
    assert 0.1 < ratio < 10.0, f"能量比异常={ratio:.3f}(残差/重编码尺度失控)"
    return f"8块后能量比={ratio:.3f} 有界"


def test_phi_zero_reduces_to_skip():
    # φ=0 时 F=纯传播(U)，merged=(传播(U)+U)/√2；这里只验证可运行且确定性
    blk = ResidualDPUBlock(size=32, n_phase=1).double()
    blk.phi.data.zero_()
    U = torch.randn(32, 32, dtype=torch.complex128)
    o1, o2 = blk(U), blk(U)
    assert torch.allclose(o1, o2), "同输入应确定性输出"
    assert torch.isfinite(o1).all()
    return "φ=0 确定性且有限"


TESTS = [test_only_phi_trainable, test_forward_and_grad,
         test_residual_skip_energy_stable, test_phi_zero_reduces_to_skip]

if __name__ == "__main__":
    n = 0
    for t in TESTS:
        try:
            print(f"  ✅ {t.__name__:<32} {t()}"); n += 1
        except Exception as e:
            print(f"  ❌ {t.__name__:<32} {type(e).__name__}: {e}")
    print(f"\nStep6(DPUResNet): {n}/{len(TESTS)} 通过")
    sys.exit(0 if n == len(TESTS) else 1)
