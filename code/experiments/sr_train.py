"""
Phase C 超分辨训练（Step5 冒烟 = DPUStack；Step6 = DPUResNet）。

全光合规：唯一可训练参数=相位掩膜φ；输出直接读探测器强度；无可训练电子层。
输出归一化：相位-only 网络无损但重分布能量，输出与 GT 有全局尺度差；
  训练用"逐图除以max"(参数自由)对齐 [0,1]；评估用 Qiao 线性重标定(更公平)。
目标：超过衍射受限输入 floor —— floor 在每个划分上**内联实算**(WF vs GT, 同 evaluate 口径)，
  不再写死常量(此前写死的 22.9 是错的：测试 floor 实为 22.4-22.7, 验证 floor 仅 ~20.8)。

用法：
  python experiments/sr_train.py --model dpuresnet --blocks 6 --z 50e-6 --epochs 40 --device cuda
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from optics.dpu import DPUStack  # noqa: E402
from datasets.biosr import BioSRPairs, BioSRTestLevel, list_test_levels, BIOSR_ROOT  # noqa: E402
from metrics.metrics import evaluate, sr_loss  # noqa: E402
from configs.config import EPS, Z_DEFAULT  # noqa: E402


def norm_out(o):
    """逐图除以 max 归一化到 ~[0,1]（参数自由的固定读出归一化）。o:[B,1,H,W]。"""
    return o / o.amax(dim=(-1, -2), keepdim=True).clamp(min=EPS)


def build_model(name, blocks, z, n_phase=1, shortcut="sharp", linear=False):
    if name == "dpustack":
        return DPUStack(size=128, n_blocks=blocks, n_phase=n_phase, z=z, nonlinear=not linear)
    if name == "dpuresnet":
        from optics.dpu_resnet import DPUResNet
        return DPUResNet(size=128, n_blocks=blocks, n_phase=n_phase, z=z, shortcut=shortcut, linear=linear)
    raise ValueError(name)


@torch.no_grad()
def eval_loader(model, loader, dev, max_batches=None):
    model.eval()
    agg = {"PSNR": 0, "SSIM": 0, "MS_SSIM": 0, "NRMSE": 0}
    nb = 0
    for bi, (wf, gt) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        out = model(wf.squeeze(1).to(dev))[:, None]            # [B,1,H,W]
        d = evaluate(out.cpu(), gt, rescale=True)
        for k in agg:
            agg[k] += d[k]
        nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


@torch.no_grad()
def floor_loader(loader, max_batches=None):
    """衍射受限输入 floor：预测=WF 本身，走与模型完全相同的 evaluate(rescale=True)。
    这样 model 与 floor 同样本、同口径，'超 floor' 的对比才公平、可自审计。"""
    agg = {"PSNR": 0, "SSIM": 0, "MS_SSIM": 0, "NRMSE": 0}
    nb = 0
    for bi, (wf, gt) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        d = evaluate(wf, gt, rescale=True)
        for k in agg:
            agg[k] += d[k]
        nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dpustack", choices=["dpustack", "dpuresnet"])
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--n_phase", type=int, default=1, help="每块探测前相位层数(块内相干混合)")
    ap.add_argument("--z", type=float, default=Z_DEFAULT)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--ssim_w", type=float, default=0.1, help="loss=MSE+ssim_w*(1-SSIM)")
    ap.add_argument("--seed", type=int, default=0, help="φ随机初始化种子(>0时固定,用于多种子确认)")
    ap.add_argument("--shortcut", default="sharp", choices=["sharp", "prop"],
                    help="残差捷径: sharp=4f锐利锚点+剥piston(好) | prop=自由传播(消谐振但模糊)")
    ap.add_argument("--linear", action="store_true", help="|U|²消融:关块间探测,整网线性(应学不了SR)")
    ap.add_argument("--train_subset", type=int, default=0, help="0=全部")
    ap.add_argument("--tag", default="", help="checkpoint 命名前缀; 空则自动按配置+jobid生成")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/fs04/scratch2/mz32/wuyujun/results")
    ap.add_argument("--data_root", default=BIOSR_ROOT,
                    help="BioSR 结构根目录(默认 Microtubules); 换结构验证泛化用, 如 .../BioSR/CCPs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    if args.seed:
        torch.manual_seed(args.seed)        # 固定 φ 初始化(须在 build_model 前)

    # 每个 run 唯一 checkpoint 名，避免并发/先后 run 互相覆盖同一文件(含结构名防跨结构冲突)
    struct = os.path.basename(args.data_root.rstrip("/"))
    tag = args.tag or (f"{struct}_{args.model}_z{int(args.z*1e6)}_b{args.blocks}_np{args.n_phase}"
                       f"_s{args.seed}_{os.environ.get('SLURM_JOB_ID', 'local')}")
    ckpt_path = os.path.join(args.out, f"{tag}_best.pt")
    print(f"[data] {args.data_root} (结构={struct})")

    tr_ds = BioSRPairs("training", root=args.data_root)
    if args.train_subset > 0:
        tr_ds = Subset(tr_ds, range(min(args.train_subset, len(tr_ds))))
    va_ds = Subset(BioSRPairs("validate", root=args.data_root), range(256))
    tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=6, drop_last=True)
    va = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    model = build_model(args.model, args.blocks, args.z, args.n_phase, args.shortcut, args.linear).to(dev)
    assert all("phi" in n for n, p in model.named_parameters() if p.requires_grad), "出现非φ可训练参数!"
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cfg] model={args.model} blocks={args.blocks} n_phase={args.n_phase} z={args.z*1e6:.0f}um "
          f"shortcut={args.shortcut} epochs={args.epochs} batch={args.batch} lr={args.lr} ssim_w={args.ssim_w} "
          f"train_n={len(tr_ds)} device={dev} φ参数={n_train:,}")
    print(f"[ckpt] {ckpt_path}")

    # 内联实算验证集 floor（同样本、同 evaluate 口径）—— 取代写死的魔法数字
    vf = floor_loader(va)
    print(f"[val floor] WF vs GT 同样本: PSNR={vf['PSNR']:.2f} SSIM={vf['SSIM']:.3f} "
          f"MS-SSIM={vf['MS_SSIM']:.3f} NRMSE={vf['NRMSE']:.4f}  (模型应超过此值)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for wf, gt in tr:
            wf, gt = wf.to(dev), gt.to(dev)
            opt.zero_grad()
            out = norm_out(model(wf.squeeze(1))[:, None])      # [B,1,H,W]
            loss = sr_loss(out, gt, ssim_w=args.ssim_w)
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()
        m = eval_loader(model, va, dev)
        flag = ""
        if m["PSNR"] > best:
            best = m["PSNR"]
            torch.save(model.state_dict(), ckpt_path)
            flag = " *best"
        print(f"  ep {ep:2d}/{args.epochs} loss={run/len(tr):.4f} | "
              f"val PSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.3f} MS-SSIM={m['MS_SSIM']:.3f} "
              f"NRMSE={m['NRMSE']:.4f}{flag} ({time.time()-t0:.0f}s)")

    # best 相对验证集自身 floor 判定（不再对比写死的测试 floor）
    print(f"\n[best val PSNR] {best:.2f} dB  (val floor {vf['PSNR']:.2f}) -> "
          f"{'✅ 超过val floor' if best > vf['PSNR'] else '⚠️ 未超val floor'} "
          f"(Δ={best - vf['PSNR']:+.2f}dB)")

    # 加载本 run 自己的 checkpoint，并断言 φ 张量数==block数（防被别的 run 覆盖）
    sd = torch.load(ckpt_path)
    n_phi = sum("phi" in k for k in sd)
    assert n_phi == args.blocks, f"checkpoint φ张量数={n_phi}!=blocks={args.blocks}，疑被其他run覆盖!"
    model.load_state_dict(sd)

    # 最终：按 SNR 等级分层评估测试集，每级内联实算同样本 floor 并报告 Δ 增益
    print("\n=== 测试集按 SNR 分层 (model vs 同样本floor; Δ=净增益) ===")
    print(f"{'level':>10} {'PSNR':>7} {'ΔPSNR':>6} {'SSIM':>7} {'ΔSSIM':>7} "
          f"{'MS-SSIM':>8} {'ΔMS':>7} {'NRMSE':>7}")
    for lv in list_test_levels(args.data_root):
        ds = Subset(BioSRTestLevel(lv, root=args.data_root), range(64))
        ld = DataLoader(ds, batch_size=args.batch, num_workers=4)
        m = eval_loader(model, ld, dev)
        f = floor_loader(ld)
        print(f"{lv:>10} {m['PSNR']:7.2f} {m['PSNR']-f['PSNR']:+6.2f} "
              f"{m['SSIM']:7.3f} {m['SSIM']-f['SSIM']:+7.3f} "
              f"{m['MS_SSIM']:8.3f} {m['MS_SSIM']-f['MS_SSIM']:+7.3f} {m['NRMSE']:7.4f}")


if __name__ == "__main__":
    main()
