"""
Step 3 / 安全门 2：MNIST 光学分类自检。

目的：证明这套衍射光学网络能被梯度下降端到端训起来（与超分辨无关，纯粹验证管线/梯度）。
若能从 10% 随机猜显著上升到高准确率，说明 asm 传播 + DPU + |U|² + φ 训练全链路正确。

架构（全光合规）：
  MNIST -> 上采样到 grid -> 振幅编码 -> DPUStack(末端读 |U|²) -> 10 个固定探测区域能量 -> argmax
  唯一可训练参数 = 相位掩膜 φ。探测区域/归一化/温度均为固定常数，无可训练电子层。
  训练用交叉熵（仅训练期的损失，非推理网络）。

用法：
  python experiments/mnist_sanity.py --epochs 20 --grid 128 --blocks 5 --device cuda
  python experiments/mnist_sanity.py --smoke           # 登录节点快速冒烟(CPU,小规模)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

from optics.dpu import DPUStack  # noqa: E402
from configs.config import WAVELENGTH, PIXEL_PITCH, Z_DEFAULT, EPS  # noqa: E402

DATA_ROOT = "/fs04/scratch2/mz32/wuyujun/data"


def build_regions(grid, zone_frac=0.12):
    """10 个固定探测区域（2 行 x 5 列），返回 [10,grid,grid] 0/1 掩膜（非可训练）。"""
    masks = torch.zeros(10, grid, grid)
    zs = max(4, int(grid * zone_frac))
    rows = [int(grid * 0.30), int(grid * 0.70)]
    cols = [int(grid * (0.12 + 0.19 * i)) for i in range(5)]
    k = 0
    for r in rows:
        for c in cols:
            r0, c0 = max(0, r - zs // 2), max(0, c - zs // 2)
            masks[k, r0:r0 + zs, c0:c0 + zs] = 1.0
            k += 1
    return masks


class DPUClassifier(nn.Module):
    """DPUStack + 固定区域读出。唯一可训练参数仍只有 φ。"""

    def __init__(self, grid=128, n_blocks=5, z=Z_DEFAULT, nonlinear=True, temperature=30.0):
        super().__init__()
        self.stack = DPUStack(size=grid, n_blocks=n_blocks, n_phase=1, z=z,
                              dx=PIXEL_PITCH, wavelength=WAVELENGTH, nonlinear=nonlinear)
        self.register_buffer("regions", build_regions(grid))  # 固定，非参数
        self.T = temperature

    def forward(self, img):                       # img: [B,grid,grid] in [0,1]
        I = self.stack(img)                       # 末端强度 [B,grid,grid]
        region_e = torch.einsum("bhw,khw->bk", I, self.regions)   # [B,10]
        total = I.sum(dim=(-1, -2)) + EPS
        return self.T * region_e / total[:, None]  # 归一化区域能量 -> logits


def get_loaders(grid, batch, smoke):
    tf = transforms.Compose([transforms.Resize(grid), transforms.ToTensor()])
    train = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf)
    test = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf)
    if smoke:
        train = torch.utils.data.Subset(train, range(256))
        test = torch.utils.data.Subset(test, range(256))
    nw = 2 if smoke else 4
    return (torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=nw),
            torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False, num_workers=nw))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.squeeze(1).to(device)
        logits = model(x)
        correct += (logits.argmax(1).cpu() == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--z", type=float, default=Z_DEFAULT)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--nonlinear", type=int, default=1, help="1=DPU带|U|²非线性, 0=被动相干D2NN")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.grid, args.blocks, args.batch = 2, 48, 3, 32

    dev = torch.device(args.device)
    print(f"[cfg] grid={args.grid} blocks={args.blocks} z={args.z*1e6:.0f}um "
          f"nonlinear={bool(args.nonlinear)} lr={args.lr} batch={args.batch} "
          f"epochs={args.epochs} device={dev}")

    model = DPUClassifier(args.grid, args.blocks, args.z, bool(args.nonlinear)).to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert all("stack" in n and "phi" in n for n, p in model.named_parameters() if p.requires_grad), \
        "出现非φ可训练参数，违反全光约束"
    print(f"[cfg] 可训练参数(全是φ): {n_train:,}")

    tr, te = get_loaders(args.grid, args.batch, args.smoke)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.5)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in tr:
            x, y = x.squeeze(1).to(dev), y.to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()
        acc = evaluate(model, te, dev)
        best = max(best, acc)
        print(f"  epoch {ep:2d}/{args.epochs}  loss={run/len(tr):.4f}  test_acc={acc:.2f}%  "
              f"best={best:.2f}%  ({time.time()-t0:.1f}s)")

    print(f"\n[结果] 最佳测试准确率 = {best:.2f}%  (随机猜=10%)")
    gate = 85.0 if not args.smoke else 15.0
    ok = best >= gate
    print(f"[安全门2] {'✅ 通过' if ok else '❌ 未达标'} (门槛 {gate:.0f}%)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
