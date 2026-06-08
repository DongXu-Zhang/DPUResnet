"""
DFCAN 电子上界基线训练 —— 与光学 DPUResNet 同数据、同指标、同 floor 口径对标。

公平性：用相同 BioSRPairs/BioSRTestLevel、相同 evaluate(rescale=True)、相同 sr_loss、
相同的逐 SNL 分层测试 + 内联实算同样本 floor。唯一区别=模型(电子 DFCAN vs 光学 DPU)。

用法：python baselines/dfcan_train.py --epochs 80 --lr 1e-3 --device cuda
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from baselines.dfcan import DFCAN  # noqa: E402
from datasets.biosr import BioSRPairs, BioSRTestLevel, list_test_levels, BIOSR_ROOT  # noqa: E402
from metrics.metrics import evaluate, sr_loss  # noqa: E402


@torch.no_grad()
def eval_loader(model, loader, dev):
    model.eval()
    agg = {"PSNR": 0, "SSIM": 0, "MS_SSIM": 0, "NRMSE": 0}
    nb = 0
    for wf, gt in loader:
        out = model(wf.to(dev))                        # [B,1,H,W]
        d = evaluate(out.cpu(), gt, rescale=True)
        for k in agg:
            agg[k] += d[k]
        nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


@torch.no_grad()
def floor_loader(loader):
    agg = {"PSNR": 0, "SSIM": 0, "MS_SSIM": 0, "NRMSE": 0}
    nb = 0
    for wf, gt in loader:
        d = evaluate(wf, gt, rescale=True)
        for k in agg:
            agg[k] += d[k]
        nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", type=int, default=64)
    ap.add_argument("--n_group", type=int, default=4)
    ap.add_argument("--n_fcab", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ssim_w", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/fs04/scratch2/mz32/wuyujun/results")
    ap.add_argument("--tag", default="dfcan")
    ap.add_argument("--data_root", default=BIOSR_ROOT,
                    help="BioSR 结构根目录(默认 Microtubules); 换结构验证泛化用")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    struct = os.path.basename(args.data_root.rstrip("/"))
    tag = args.tag if args.tag != "dfcan" else f"{struct}_dfcan"
    ckpt = os.path.join(args.out, f"{tag}_best.pt")
    print(f"[data] {args.data_root} (结构={struct})  [ckpt-tag] {tag}")

    tr_ds = BioSRPairs("training", root=args.data_root)
    va_ds = Subset(BioSRPairs("validate", root=args.data_root), range(256))
    tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=6, drop_last=True)
    va = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    model = DFCAN(in_ch=1, channel=args.channel, n_group=args.n_group,
                  n_fcab=args.n_fcab, scale=1).to(dev)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cfg] DFCAN channel={args.channel} n_group={args.n_group} n_fcab={args.n_fcab} "
          f"epochs={args.epochs} batch={args.batch} lr={args.lr} train_n={len(tr_ds)} "
          f"device={dev} 参数={n_param:,} (对比: 光学DPU仅98,304个φ)")
    print(f"[ckpt] {ckpt}")
    vf = floor_loader(va)
    print(f"[val floor] WF vs GT 同样本: PSNR={vf['PSNR']:.2f} SSIM={vf['SSIM']:.3f}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for wf, gt in tr:
            wf, gt = wf.to(dev), gt.to(dev)
            opt.zero_grad()
            out = model(wf)                            # [B,1,H,W]
            loss = sr_loss(out, gt, ssim_w=args.ssim_w)
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()
        m = eval_loader(model, va, dev)
        flag = ""
        if m["PSNR"] > best:
            best = m["PSNR"]
            torch.save(model.state_dict(), ckpt)
            flag = " *best"
        print(f"  ep {ep:3d}/{args.epochs} loss={run/len(tr):.4f} | "
              f"val PSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.3f} MS-SSIM={m['MS_SSIM']:.3f} "
              f"NRMSE={m['NRMSE']:.4f}{flag} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n[best val PSNR] {best:.2f} dB  (val floor {vf['PSNR']:.2f}, Δ={best-vf['PSNR']:+.2f})")
    model.load_state_dict(torch.load(ckpt))
    print("\n=== 测试集按 SNR 分层 (DFCAN vs 同样本floor; Δ=净增益) ===")
    print(f"{'level':>10} {'PSNR':>7} {'ΔPSNR':>6} {'SSIM':>7} {'ΔSSIM':>7} {'MS-SSIM':>8} {'NRMSE':>7}")
    for lv in list_test_levels(args.data_root):
        ds = Subset(BioSRTestLevel(lv, root=args.data_root), range(64))
        ld = DataLoader(ds, batch_size=args.batch, num_workers=4)
        m = eval_loader(model, ld, dev)
        f = floor_loader(ld)
        print(f"{lv:>10} {m['PSNR']:7.2f} {m['PSNR']-f['PSNR']:+6.2f} "
              f"{m['SSIM']:7.3f} {m['SSIM']-f['SSIM']:+7.3f} {m['MS_SSIM']:8.3f} {m['NRMSE']:7.4f}")


if __name__ == "__main__":
    main()
