"""
光 vs 电 可视化: WF(输入) -> 光学DPU-SR -> 电子DFCAN-SR -> GT 四联图。

显示口径与评测完全一致: 每张预测都经 linear_rescale(pred, gt) (Qiao Eq.14-15 逐图仿射),
即"被打分的那张图"本身, 不做任何额外美化。逐面板标注 PSNR/SSIM。
WF 同样也 rescale 后作为 floor 显示, 保证三者同口径。

输出: /fs04/scratch2/mz32/wuyujun/results/fig_optical_vs_electronic.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
import torch                                                  # noqa: E402

from optics.dpu_resnet import DPUResNet                       # noqa: E402
from baselines.dfcan import DFCAN                             # noqa: E402
from datasets.biosr import BioSRTestLevel, list_test_levels, BIOSR_ROOT  # noqa: E402
from metrics.metrics import linear_rescale, psnr, ssim        # noqa: E402

RES = "/fs04/scratch2/mz32/wuyujun/results"
DEV = torch.device("cpu")
CMAP = "hot"                                                  # 荧光显微常用热图LUT


def disp(pred, gt):
    """返回(被打分口径的)rescale后图 + 其PSNR/SSIM。pred/gt: [1,1,H,W]."""
    r = linear_rescale(pred, gt).clamp(0, 1)
    return r[0, 0].numpy(), psnr(r, gt).item(), ssim(r, gt).item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=BIOSR_ROOT)
    ap.add_argument("--opt_ckpt", default=os.path.join(RES, "sh_z15_b6_best.pt"))
    ap.add_argument("--dfcan_ckpt", default=os.path.join(RES, "dfcan30_best.pt"))
    ap.add_argument("--z", type=float, default=15e-6)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--idx", type=int, default=8, help="测试集固定样本索引(可复现)")
    ap.add_argument("--struct", default="microtubule", help="标题用结构名")
    ap.add_argument("--out", default=os.path.join(RES, "fig_optical_vs_electronic.png"))
    args = ap.parse_args()

    # 光学(冠军同配置)
    opt = DPUResNet(size=128, n_blocks=args.blocks, z=args.z, shortcut="sharp").to(DEV)
    opt.load_state_dict(torch.load(args.opt_ckpt, map_location=DEV))
    opt.eval()
    n_phi = sum(p.numel() for p in opt.parameters() if p.requires_grad)
    assert all("phi" in n for n, p in opt.named_parameters() if p.requires_grad), "光学网络出现非φ可训练参数!"

    # 电子上界
    ele = DFCAN(in_ch=1, channel=64, n_group=4, n_fcab=4, scale=1).to(DEV)
    ele.load_state_dict(torch.load(args.dfcan_ckpt, map_location=DEV))
    ele.eval()
    n_ele = sum(p.numel() for p in ele.parameters() if p.requires_grad)
    print(f"[{args.struct}] 光学φ={n_phi:,}  电子DFCAN={n_ele:,}  (电子 {n_ele/n_phi:.1f}×)  data={args.data_root}")

    # 三个代表性 SNR 层(自适应: 该结构最低/中/最高 level), 固定索引(可复现)
    lvls = list_test_levels(args.data_root)
    rows = [(lvls[0], args.idx, "low SNR"), (lvls[len(lvls) // 2], args.idx, "mid SNR"),
            (lvls[-1], args.idx, "high SNR")]
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)))
    col_titles = ["WF input (original)", "Optical DPU-SR (ours)\n(98K phi, diffractive)",
                  "Electronic DFCAN-SR\n(1.82M params)", "Ground Truth (SIM)"]

    for r, (lv, idx, tag) in enumerate(rows):
        ds = BioSRTestLevel(lv, root=args.data_root)
        wf, gt = ds[idx]                                       # [1,H,W]
        wf, gt = wf[None], gt[None]                            # [1,1,H,W]

        o = opt(wf.squeeze(1).to(DEV))[:, None].cpu()          # [1,1,H,W] = |U|²
        e = ele(wf.to(DEV)).cpu()

        wf_img, wf_p, wf_s = disp(wf, gt)                      # WF 同样 rescale 当 floor
        o_img, o_p, o_s = disp(o, gt)
        e_img, e_p, e_s = disp(e, gt)
        gt_img = gt[0, 0].numpy()

        panels = [
            (wf_img, f"PSNR {wf_p:.2f} / SSIM {wf_s:.3f}"),
            (o_img, f"PSNR {o_p:.2f} (+{o_p-wf_p:.2f}) / SSIM {o_s:.3f}"),
            (e_img, f"PSNR {e_p:.2f} (+{e_p-wf_p:.2f}) / SSIM {e_s:.3f}"),
            (gt_img, "reference"),
        ]
        for c, (img, sub) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=CMAP, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(sub, fontsize=9)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{lv}\n{tag}", fontsize=11)

    fig.suptitle(f"BioSR {args.struct}:  WF input  vs  Optical DPU (ours)  vs  Electronic DFCAN  vs  Ground Truth   "
                 "(same data / same metric / same floor)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
