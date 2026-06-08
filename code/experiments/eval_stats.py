"""
严谨评估工具:对某结构测试集算逐 SNR 层 Δ-over-floor,带 cell-block bootstrap 95% 置信区间。

为什么需要:测试 patch 来自少数 held-out cell(每 cell 连续 per_cell 张,强相关),
直接按 patch 算均值会低估方差。正确做法 = 按 **cell** 重采样(block bootstrap),
有效样本量 = cell 数(非 patch 数)。同时支持两模型的**配对**比较(同 patch 抵消 cell 间方差)。

口径与训练评估完全一致:每张图 evaluate(pred, gt, rescale=True)(Qiao 仿射 + clamp)。

用法:
  # 单模型逐层 Δ±CI
  python experiments/eval_stats.py --model optical --ckpt <opt.pt> --z 15e-6 \
      --data_root .../BioSR/F-actin_40k
  python experiments/eval_stats.py --model dfcan --ckpt <dfcan.pt> --data_root ...
  # 两光学模型配对比较(同测试集): Δ_new - Δ_old 的逐层均值±CI
  python experiments/eval_stats.py --model optical --ckpt <new.pt> --z 15e-6 \
      --pair_ckpt <old.pt> --data_root ...
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets.biosr import BioSRTestLevel, list_test_levels  # noqa: E402
from metrics.metrics import evaluate  # noqa: E402


def build_optical(ckpt_path, z, dev):
    """从 checkpoint 推断 block 数(phi 张量个数)并构造 DPUResNet。"""
    from optics.dpu_resnet import DPUResNet
    sd = torch.load(ckpt_path, map_location=dev)
    n_blocks = sum(1 for k in sd if k.endswith("phi"))
    net = DPUResNet(size=128, n_blocks=n_blocks, z=z, shortcut="sharp").to(dev)
    net.load_state_dict(sd)
    net.eval()
    assert all("phi" in n for n, p in net.named_parameters() if p.requires_grad), "非φ可训练!"
    return net, n_blocks


def build_dfcan(ckpt_path, dev):
    from baselines.dfcan import DFCAN
    net = DFCAN(in_ch=1, channel=64, n_group=4, n_fcab=4, scale=1).to(dev)
    net.load_state_dict(torch.load(ckpt_path, map_location=dev))
    net.eval()
    return net


@torch.no_grad()
def per_patch_delta(net, is_optical, loader, dev):
    """返回逐 patch 的 (模型PSNR, floorPSNR, 模型SSIM, floorSSIM) 四个数组(顺序=文件顺序)。
    前向按 batch 一次算完(快), 指标逐 patch 提取(配对/bootstrap 需要)。"""
    mp, fp, ms, fs = [], [], [], []
    for wf, gt in loader:
        if is_optical:
            out = net(wf.squeeze(1).to(dev))[:, None].cpu()   # 批量前向
        else:
            out = net(wf.to(dev)).cpu()
        for i in range(wf.shape[0]):
            dm = evaluate(out[i:i + 1], gt[i:i + 1], rescale=True)
            df = evaluate(wf[i:i + 1], gt[i:i + 1], rescale=True)
            mp.append(dm["PSNR"]); fp.append(df["PSNR"])
            ms.append(dm["SSIM"]); fs.append(df["SSIM"])
    return np.array(mp), np.array(fp), np.array(ms), np.array(fs)


def cell_ids(n_patches, n_cells):
    """测试 patch 按 cell 连续排列(每 cell per_cell 张)。返回每个 patch 的 cell 下标。"""
    per = int(np.ceil(n_patches / n_cells))
    return np.array([i // per for i in range(n_patches)])


def block_bootstrap_ci(values, cells, B=2000, seed=0):
    """按 cell 重采样的 block bootstrap。values: 逐patch量。返回 (均值, lo95, hi95)。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(cells)
    groups = [values[cells == c] for c in uniq]
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(groups), len(groups))   # 重采样 cell
        boot[b] = np.concatenate([groups[k] for k in idx]).mean()
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["optical", "dfcan"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pair_ckpt", default="", help="可选: 另一光学ckpt, 做配对 Δnew-Δold(同测试集)")
    ap.add_argument("--z", type=float, default=15e-6)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cells", type=int, default=12, help="测试集 held-out cell 数(默认12)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    is_opt = args.model == "optical"
    if is_opt:
        net, nb = build_optical(args.ckpt, args.z, dev)
        print(f"[optical] {os.path.basename(args.ckpt)}  blocks={nb}  z={args.z*1e6:.0f}um")
    else:
        net = build_dfcan(args.ckpt, dev)
        print(f"[dfcan] {os.path.basename(args.ckpt)}")

    net2 = None
    if args.pair_ckpt:
        net2, nb2 = build_optical(args.pair_ckpt, args.z, dev)
        print(f"[pair(old)] {os.path.basename(args.pair_ckpt)}  blocks={nb2}")

    print(f"[data] {args.data_root}  cells={args.cells}")
    print(f"\n{'level':>9} {'ΔPSNR':>7} {'95%CI':>16} {'ΔSSIM':>7} {'95%CI':>16}"
          + ("   | 配对Δ(new-old)PSNR [95%CI]" if net2 else ""))
    for lv in list_test_levels(args.data_root):
        ds = BioSRTestLevel(lv, root=args.data_root)
        ld = DataLoader(ds, batch_size=16)
        mp, fp, ms, fs = per_patch_delta(net, is_opt, ld, dev)
        cells = cell_ids(len(mp), args.cells)
        dpsnr = mp - fp
        dssim = ms - fs
        m1, lo1, hi1 = block_bootstrap_ci(dpsnr, cells)
        m2, lo2, hi2 = block_bootstrap_ci(dssim, cells)
        line = (f"{lv:>9} {m1:+7.2f} [{lo1:+5.2f},{hi1:+5.2f}] "
                f"{m2:+7.3f} [{lo2:+6.3f},{hi2:+6.3f}]")
        if net2:
            mp2, fp2, _, _ = per_patch_delta(net2, True, DataLoader(ds, batch_size=16), dev)
            paired = (mp - fp) - (mp2 - fp2)            # 同 patch 配对差(floor 抵消)
            pm, plo, phi = block_bootstrap_ci(paired, cells)
            sig = "✅" if plo > 0 else ("❌" if phi < 0 else "~")
            line += f"   | {pm:+.2f} [{plo:+.2f},{phi:+.2f}] {sig}"
        print(line)


if __name__ == "__main__":
    main()
