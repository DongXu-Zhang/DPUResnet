"""
给老师汇报用的定量图表生成器。

所有数字均**逐字摘自** code/logs/ 的实跑日志(每条标注 SLURM job ID 出处),
不重新评测、不编造。评测口径 = Qiao Eq.14-15 线性重标定后的 PSNR/SSIM,
Δ = 与"衍射受限输入 floor(WF 当预测)"的同样本净增益(>0 即真正超过宽场)。

产出(results/):
  fig_results_main.png        主结果: MT 逐 SNR 光 vs 电 vs floor  +  跨结构回收率
  fig_ablation_nonlinearity.png  ⭐|U|² 消融: 关掉平方非线性 → 学不了 SR(证明方法成立)
  fig_physics_zsweep.png      物理: ΔPSNR 随传播距离 z 的扫描 + piston-bug 修复
  fig_learned_phi.png         可解释性: 冠军网络学到的 6 层相位掩膜 φ

英文图注(老师看代码与论文均英文口径)。
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RES = "/fs04/scratch2/mz32/wuyujun/results"

# 配色(全局统一)
C_OPT = "#1f9e89"     # 光学 DPU(青绿)
C_ELE = "#e07b39"     # 电子 DFCAN(橙)
C_FLOOR = "#9aa0a6"   # floor / 关掉非线性
C_LIN = "#c0392b"     # 线性(消融)
C_OK = "#2e7d32"      # 修复后

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 150, "savefig.dpi": 150})

# =========================================================================
# 经日志核验的真实数据(ΔPSNR over floor;source = code/logs/<file>)
# =========================================================================

# ---- Microtubule (主结构) floor PSNR=20.85, 9 levels ----
MT_L = np.arange(1, 10)
MT_OPT = [1.69, 2.55, 2.92, 3.18, 3.34, 3.41, 3.44, 3.54, 3.59]   # sr_sh_z15_b6_56061821 (φ=98,304)
MT_DFCAN = [4.10, 4.58, 4.88, 5.31, 5.53, 5.56, 5.62, 5.77, 5.86]  # sr_dfcan30_56066145 (1.82M)

# ---- |U|² 消融 (MT, 同 b6 同 z, 只切换块间平方探测) ----
MT_NL_Z15 = MT_OPT                                                  # 非线性 z15 = 冠军
MT_LIN_Z15 = [1.08, 1.36, 1.46, 1.56, 1.60, 1.61, 1.61, 1.64, 1.66]  # sr_abl_lin_z15_b6_56064401
MT_LIN_Z30 = [-0.18, -0.09, -0.07, -0.02, -0.01, 0.00, -0.01, -0.01, -0.01]  # sr_abl_lin_z30_b6_56064402
# L9 汇总条形 (非线性 vs 线性 @ z15/z30)
ABL_NL = {"z15": 3.59, "z30": 2.44}    # 非线性: sh_z15_b6 / sh_z30_56059393
ABL_LIN = {"z15": 1.66, "z30": -0.01}  # 线性:   abl_lin_z15 / abl_lin_z30

# ---- z 扫描 (MT, sharp 捷径, b6, 单层深度) ΔPSNR@L9 ----
Z_VALS = [10, 12, 15, 20, 25, 30, 40, 45, 50]
Z_DPSNR = [2.40, 3.42, 3.59, 3.23, 2.81, 2.44, 1.61, 1.27, 1.00]
# job: sh_z10_b6_56063628 / z12_56063629 / z15_56061821 / z20_56059391 / z25_56059392
#      z30_56059393 / z40_56059394 / z45_56059395 / z50_56059396
# piston-bug: 普通(prop)捷径 vs sharp 捷径 @ z=45µm
PROP_Z45 = -2.89   # sr_rn_z45_56051406 (无 sharp,piston 谐振崩溃)
SHARP_Z45 = 1.27   # sr_sh_z45_56059395 (sharp 修复)

# ---- 跨结构(各自最佳 SNR level 的 Δ) ----
ST_NAMES = ["Microtubule\n(L9)", "F-actin\n(40k, L9)", "ER\n(40k, L6)"]
ST_OPT = [3.59, 2.11, 2.80]     # sh_z15_b6 / f40_b6(56095321) / e40_b12(56095324)
ST_DFCAN = [5.86, 4.87, 6.40]   # dfcan30 / 56095880 / 56095881
ST_REC = [o / e * 100 for o, e in zip(ST_OPT, ST_DFCAN)]  # 回收率 %

# ---- 数据量消融(光学,12k→40k)----
DS_NAMES = ["F-actin (L9)", "ER (L6)"]
DS_12K = [0.66, 1.73]   # 56090173 / 56090837
DS_40K = [2.11, 2.80]   # 56095321 / 56095324


# =========================================================================
def fig_main():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

    # (a) MT 逐 SNR: 光 vs 电 vs floor
    a = ax[0]
    a.axhline(0, color=C_FLOOR, ls="--", lw=1.4, label="WF floor (diffraction-limited input)")
    a.plot(MT_L, MT_DFCAN, "-o", color=C_ELE, lw=2, ms=5, label="Electronic DFCAN (1.82M params)")
    a.plot(MT_L, MT_OPT, "-o", color=C_OPT, lw=2, ms=5, label="Optical DPU-ResNet (98K φ, ours)")
    a.fill_between(MT_L, 0, MT_OPT, color=C_OPT, alpha=0.10)
    a.set_xlabel("SNR level (1 = noisiest → 9 = cleanest)")
    a.set_ylabel("ΔPSNR over floor (dB)")
    a.set_title("(a) Microtubule: gain over wide-field, by SNR")
    a.set_xticks(MT_L)
    a.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    a.grid(alpha=0.25)
    a.annotate(f"+{MT_OPT[-1]:.2f} dB", (9, MT_OPT[-1]), textcoords="offset points",
               xytext=(-4, 6), color=C_OPT, fontweight="bold", ha="right")
    a.annotate(f"+{MT_DFCAN[-1]:.2f} dB", (9, MT_DFCAN[-1]), textcoords="offset points",
               xytext=(-4, 6), color=C_ELE, fontweight="bold", ha="right")

    # (b) 跨结构回收率
    b = ax[1]
    x = np.arange(len(ST_NAMES))
    w = 0.38
    b.bar(x - w / 2, ST_OPT, w, color=C_OPT, label="Optical DPU (ours)")
    b.bar(x + w / 2, ST_DFCAN, w, color=C_ELE, label="Electronic DFCAN")
    for i in range(len(x)):
        b.text(x[i] - w / 2, ST_OPT[i] + 0.08, f"+{ST_OPT[i]:.2f}", ha="center", fontsize=8.5, color=C_OPT)
        b.text(x[i] + w / 2, ST_DFCAN[i] + 0.08, f"+{ST_DFCAN[i]:.2f}", ha="center", fontsize=8.5, color=C_ELE)
        b.text(x[i], -0.55, f"optical recovers\n{ST_REC[i]:.0f}% of electronic", ha="center",
               fontsize=8, color="#333")
    b.set_xticks(x)
    b.set_xticklabels(ST_NAMES)
    b.set_ylabel("ΔPSNR over floor (dB)")
    b.set_title("(b) Same-grid SR across 3 structures (best SNR level)")
    b.set_ylim(-1.0, 7.6)
    b.legend(loc="upper center", ncol=2, fontsize=8.5, framealpha=0.95)
    b.grid(axis="y", alpha=0.25)
    b.axhline(0, color="k", lw=0.8)

    fig.suptitle("All-optical DPU super-resolution on BioSR  (Δ = gain over diffraction-limited wide-field; same data / same metric)",
                 fontsize=10.5, y=1.00)
    fig.text(0.5, -0.02,
             "Honest negative — CCPs (sparse puncta): BOTH optical (Δ≈0.0) and DFCAN (failed to train, stays below floor) fail to beat the wide-field floor, so CCPs is omitted from panel (b).",
             ha="center", fontsize=7.6, color=C_LIN)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = os.path.join(RES, "fig_results_main.png")
    fig.savefig(out, bbox_inches="tight")
    print("[saved]", out)


def fig_ablation():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

    # (a) 逐 SNL: 非线性 vs 线性
    a = ax[0]
    a.axhline(0, color=C_FLOOR, ls="--", lw=1.4, label="WF floor")
    a.plot(MT_L, MT_NL_Z15, "-o", color=C_OPT, lw=2, ms=5, label="|U|² ON  (nonlinear, z=15µm) — full DPU")
    a.plot(MT_L, MT_LIN_Z15, "-s", color=C_LIN, lw=2, ms=5, label="|U|² OFF (linear, z=15µm)")
    a.plot(MT_L, MT_LIN_Z30, "-^", color="#7f8c8d", lw=1.8, ms=5, label="|U|² OFF (linear, z=30µm) → collapses to floor")
    a.set_xlabel("SNR level")
    a.set_ylabel("ΔPSNR over floor (dB)")
    a.set_title("(a) Remove the |U|² detection → the net becomes linear")
    a.set_xticks(MT_L)
    a.legend(loc="upper left", fontsize=8.2, framealpha=0.95)
    a.grid(alpha=0.25)

    # (b) L9 汇总条形
    b = ax[1]
    groups = ["z = 15 µm", "z = 30 µm"]
    x = np.arange(2)
    w = 0.38
    nl = [ABL_NL["z15"], ABL_NL["z30"]]
    lin = [ABL_LIN["z15"], ABL_LIN["z30"]]
    b.bar(x - w / 2, nl, w, color=C_OPT, label="|U|² ON (nonlinear)")
    b.bar(x + w / 2, lin, w, color=C_FLOOR, label="|U|² OFF (linear)")
    for i in range(2):
        b.text(x[i] - w / 2, nl[i] + 0.06, f"+{nl[i]:.2f}", ha="center", fontsize=9, color=C_OPT, fontweight="bold")
        yl = lin[i]
        b.text(x[i] + w / 2, yl + (0.06 if yl >= 0 else -0.20), f"{yl:+.2f}", ha="center",
               fontsize=9, color=C_LIN, fontweight="bold")
    b.set_xticks(x)
    b.set_xticklabels(groups)
    b.set_ylabel("ΔPSNR over floor @ best level (dB)")
    b.set_title("(b) Nonlinearity ≈ doubles the gain; at z=30µm linear dies")
    b.axhline(0, color="k", lw=0.8)
    b.set_ylim(-0.6, 4.2)
    b.legend(loc="upper right", fontsize=8.5)
    b.grid(axis="y", alpha=0.25)
    b.annotate("linear collapses\nto the floor", (1 + w / 2, -0.01), textcoords="offset points",
               xytext=(6, 22), fontsize=8, color=C_LIN,
               arrowprops=dict(arrowstyle="->", color=C_LIN, lw=1.2))

    fig.suptitle("|U|^2 ablation — the square-law detection is the ONLY nonlinearity, and it is essential for SR "
                 "(confirms supervisor reminder 3)", fontsize=10, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RES, "fig_ablation_nonlinearity.png")
    fig.savefig(out, bbox_inches="tight")
    print("[saved]", out)


def fig_zsweep():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.1, 1]})

    # (a) ΔPSNR vs z
    a = ax[0]
    a.axhline(0, color=C_FLOOR, ls="--", lw=1.2)
    a.plot(Z_VALS, Z_DPSNR, "-o", color=C_OPT, lw=2, ms=6)
    i_best = int(np.argmax(Z_DPSNR))
    a.plot(Z_VALS[i_best], Z_DPSNR[i_best], "o", color="#b8860b", ms=12, mfc="none", mew=2.2)
    a.annotate(f"champion\nz=15µm  +{Z_DPSNR[i_best]:.2f} dB", (Z_VALS[i_best], Z_DPSNR[i_best]),
               textcoords="offset points", xytext=(14, -2), fontsize=8.5, color="#7a5c00")
    a.set_xlabel("inter-layer propagation distance z (µm)")
    a.set_ylabel("ΔPSNR over floor (dB)")
    a.set_title("(a) Imaging needs SMALL z: strong diffraction (large z) scrambles structure")
    a.grid(alpha=0.25)
    a.text(30, 3.2, "small z → field stays aligned\n(deconvolution regime)", fontsize=7.8, color="#555")
    a.text(42, 0.6, "large z → over-mixing,\ngain decays", fontsize=7.8, color="#555")

    # (b) piston-bug fix @ z45
    b = ax[1]
    bars = b.bar(["prop\nshortcut", "sharp\nshortcut\n(fix)"], [PROP_Z45, SHARP_Z45],
                 color=[C_LIN, C_OK], width=0.6)
    b.axhline(0, color="k", lw=0.8)
    for rect, v in zip(bars, [PROP_Z45, SHARP_Z45]):
        b.text(rect.get_x() + rect.get_width() / 2, v + (0.12 if v >= 0 else -0.30),
               f"{v:+.2f}", ha="center", fontsize=9.5, fontweight="bold",
               color=(C_OK if v >= 0 else C_LIN))
    b.set_ylabel("ΔPSNR over floor @ z=45µm (dB)")
    b.set_title("(b) Piston-resonance bug\nfixed by 4f sharp shortcut")
    b.set_ylim(-3.6, 2.2)
    b.grid(axis="y", alpha=0.25)

    fig.suptitle("Physical tuning of the DPU: propagation distance z, and the piston-carrier resonance we found & fixed",
                 fontsize=10, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RES, "fig_physics_zsweep.png")
    fig.savefig(out, bbox_inches="tight")
    print("[saved]", out)


def fig_phi():
    """加载冠军 ckpt,可视化学到的相位掩膜 φ(对光学审稿人最直观)。"""
    import torch
    ckpt = os.path.join(RES, "sh_z15_b6_best.pt")
    sd = torch.load(ckpt, map_location="cpu")
    phis = []
    for k, v in sd.items():
        if "phi" in k.lower() and hasattr(v, "ndim") and v.numel() >= 128 * 128:
            t = v.detach().float().squeeze()
            if t.ndim == 3:        # [n_phase,H,W]
                for j in range(t.shape[0]):
                    phis.append((f"{k}[{j}]", t[j]))
            elif t.ndim == 2:
                phis.append((k, t))
    if not phis:
        print("[warn] no phi params found; keys=", list(sd.keys())[:8])
        return
    # 减去各自均值看空间调制;共享对称色标(全局稳健幅度),显出从浅到深的渐进结构
    devs = [(name, (t.numpy() - float(t.numpy().mean()))) for name, t in phis]
    vmax = float(np.percentile(np.abs(np.concatenate([d.ravel() for _, d in devs])), 99))
    n = len(devs)
    cols = min(n, 6)
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 3.0 * rows))
    ax = np.atleast_2d(ax)
    for idx, (name, d) in enumerate(devs):
        r, c = divmod(idx, cols)
        a = ax[r, c]
        im = a.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ptp = float(d.max() - d.min())
        a.set_title(f"block {idx + 1}   (φ range {ptp:.2f} rad)", fontsize=8.5)
        a.set_xticks([]); a.set_yticks([])
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        ax[r, c].axis("off")
    cb = fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.025, pad=0.02)
    cb.set_label("phase deviation from mean (rad)")
    fig.suptitle("Learned phase masks φ of the champion optical DPU-ResNet — the ONLY trainable parameters (6 blocks × 128×128).\n"
                 "Modulation is smooth and gentle (a few rad p-p), growing deeper down the stack — SLM-realizable.",
                 fontsize=9.5, y=1.02)
    out = os.path.join(RES, "fig_learned_phi.png")
    fig.savefig(out, bbox_inches="tight")
    print("[saved]", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "main"):
        fig_main()
    if which in ("all", "abl"):
        fig_ablation()
    if which in ("all", "z"):
        fig_zsweep()
    if which in ("all", "phi"):
        fig_phi()
    print("done.")
