"""Visualize the ACTUAL BioSR microtubule data used in the wuyujun experiments."""
import os, numpy as np, tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/fs04/scratch2/mz32/wuyujun/data/BioSR/Microtubules"
OUT = "/fs04/scratch2/mz32/wuyujun/biosr_demo"
os.makedirs(OUT, exist_ok=True)

def load(p):
    a = tifffile.imread(p).astype(np.float32)
    return a[..., 0] if a.ndim == 3 else a

def pnorm(x, lo=0.1, hi=99.9):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-7), 0, 1)

scene = "001.tif"
levels = sorted(os.listdir(f"{R}/test_wf"))
gt = pnorm(load(f"{R}/test_gt/{scene}"))
wfs = {lv: pnorm(load(f"{R}/test_wf/{lv}/{scene}")) for lv in levels}

# ---------------- FIG A: same scene across 9 SNR levels + GT target
fig, ax = plt.subplots(2, 5, figsize=(15, 6.4))
order = levels  # level_01 .. level_09
for k, lv in enumerate(order):
    a = ax.ravel()[k]
    a.imshow(wfs[lv], cmap="gray", vmin=0, vmax=1)
    tag = "  <- fewest photons\n     noisiest, HARDEST input" if k == 0 else (
          "  <- most photons, cleanest" if k == 8 else "")
    a.set_title(f"WF  {lv}{tag}", fontsize=9)
    a.set_xticks([]); a.set_yticks([])
# 10th panel = GT
ag = ax.ravel()[9]
ag.imshow(gt, cmap="gray", vmin=0, vmax=1)
ag.set_title("GT-SIM (TARGET / answer)\nultra-high SNR, high-res", fontsize=9, color="crimson")
ag.set_xticks([]); ag.set_yticks([])
for s in ag.spines.values(): s.set_color("crimson"); s.set_linewidth(2.5)
fig.suptitle("BioSR microtubules — ONE scene, 9 SNR levels of input (WF) + the single GT target.  All 128x128, same grid.\n"
             "Network INPUT = a WF (any level).  TARGET it must produce = the GT.  Evaluate per-level to see denoise+SR vs SNR.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(f"{OUT}/A_snr_levels_and_gt.png", dpi=110); plt.close(fig)

# ---------------- FIG B: the task — WF input vs GT target (+ profile) on SAME grid
lv_lo, lv_hi = "level_01", "level_09"
wf_lo, wf_hi = wfs[lv_lo], wfs[lv_hi]
row = 64
fig = plt.figure(figsize=(14, 7))
panels = [(wf_lo, f"INPUT example: WF {lv_lo}\n(blurry + noisy, low SNR)"),
          (wf_hi, f"WF {lv_hi}\n(cleaner, but STILL blurry)"),
          (gt,    "TARGET: GT-SIM\n(sharp, high-res)")]
for j, (im, t) in enumerate(panels):
    a = fig.add_subplot(2, 3, j + 1)
    a.imshow(im, cmap="gray", vmin=0, vmax=1); a.set_title(t, fontsize=10)
    a.axhline(row, color="yellow", lw=0.8, alpha=0.7)
    a.set_xticks([]); a.set_yticks([])
axp = fig.add_subplot(2, 1, 2)
axp.plot(wf_lo[row], label="WF level_01 (input)", color="tab:blue", alpha=0.8)
axp.plot(wf_hi[row], label="WF level_09", color="tab:green", alpha=0.8)
axp.plot(gt[row],    label="GT (target)", color="crimson", lw=2)
axp.set_title(f"Brightness along the yellow line (row {row}):  GT separates close filaments that WF blurs into one", fontsize=10)
axp.set_xlabel("pixel"); axp.legend(fontsize=9); axp.set_xlim(0, 127)
fig.suptitle("YOUR EXPERIMENT in one picture:  input = WF (128x128, low-SNR, diffraction-blurred)  ->  output must match GT (128x128, sharp).\n"
             "Same grid (no 2x upscaling here). Beat the WF->GT 'floor':  PSNR~22.6-22.9 dB,  SSIM~0.645-0.667.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(f"{OUT}/B_task_wf_to_gt.png", dpi=110); plt.close(fig)

print("done ->", OUT)
print("scene:", scene, "| GT", gt.shape, "| levels:", len(levels))
