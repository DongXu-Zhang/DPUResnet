"""Visualize spatial<->frequency domain, the diffraction 'circle', resolution, and SIM."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = "/fs04/scratch2/mz32/wuyujun/fourier_demo"
N = 256
yy, xx = np.mgrid[0:N, 0:N]
cx = cy = N / 2.0

def spectrum(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    return F

def logmag(F):
    return np.log1p(np.abs(F))

def lowpass(img, R):
    F = spectrum(img)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = (rr <= R).astype(float)
    Fm = F * mask
    out = np.real(np.fft.ifft2(np.fft.ifftshift(Fm)))
    return out, Fm, mask

def lowpass_soft(img, sigma):
    F = spectrum(img)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    g = np.exp(-(rr ** 2) / (2 * sigma ** 2))
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * g)))

def norm(a):
    a = a - a.min()
    return a / (a.max() + 1e-9)

# ---------------------------------------------------------------- FIG 1: stripe -> dots
def stripe(fx, fy):
    return 0.5 + 0.5 * np.cos(2 * np.pi * (fx * (xx - cx) + fy * (yy - cy)) / N)

cases = [("coarse horizontal\nstripe", 0, 6),
         ("fine horizontal\nstripe",   0, 28),
         ("fine vertical\nstripe",     28, 0),
         ("fine diagonal\nstripe",     20, 20)]
W = 55
fig, ax = plt.subplots(2, 4, figsize=(14, 7.4))
for j, (name, fx, fy) in enumerate(cases):
    img = stripe(fx, fy)
    ax[0, j].imshow(img, cmap="gray"); ax[0, j].set_title(name, fontsize=11)
    M = logmag(spectrum(img))
    ax[1, j].imshow(M, cmap="magma", vmin=0, vmax=M.max())
    for (px, py) in [(cx + fx, cy + fy), (cx - fx, cy - fy)]:
        ax[1, j].add_patch(Circle((px, py), 6, fill=False, color="cyan", lw=2.5))
    ax[1, j].plot(cx, cy, "w+", ms=8)
    ax[1, j].set_xlim(cx - W, cx + W); ax[1, j].set_ylim(cy + W, cy - W)
    ax[1, j].set_title("frequency map (zoom-in)", fontsize=10)
    for a in (ax[0, j], ax[1, j]):
        a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_ylabel("SPACE domain\n(what you see)", fontsize=11)
ax[1, 0].set_ylabel("FREQUENCY map\n(cyan = this stripe's\ntwo points; +=center)", fontsize=10)
fig.suptitle("FIG 1   One stripe  =  one pair of points on the frequency map\n"
             "coarser stripe -> point closer to center   |   finer stripe -> point farther out   |   "
             "stripe direction -> point direction", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(f"{OUT}/01_stripe_to_dots.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- FIG 2: scene + spectrum
scene = np.zeros((N, N))
# rings (CCP-like)
for (oy, ox, R) in [(70, 70, 14), (60, 160, 10), (180, 90, 12)]:
    rr = np.sqrt((xx - ox) ** 2 + (yy - oy) ** 2)
    scene += np.exp(-((rr - R) ** 2) / (2 * 2.0 ** 2))
# filaments (lines)
for (y0, x0, y1, x1) in [(120, 30, 150, 220), (200, 40, 210, 230), (40, 200, 230, 210)]:
    t = np.linspace(0, 1, 600)
    ys = (y0 + t * (y1 - y0)).astype(int); xs = (x0 + t * (x1 - x0)).astype(int)
    for dx in range(-1, 2):
        scene[np.clip(ys, 0, N-1), np.clip(xs+dx, 0, N-1)] += 1.0
scene = norm(scene)
fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
ax[0].imshow(scene, cmap="gray"); ax[0].set_title("A picture (space domain)\nrings + filaments, like a microscopy image", fontsize=11)
ax[1].imshow(logmag(spectrum(scene)), cmap="magma"); ax[1].set_title("Its frequency map (Fourier transform)\nbright center=coarse, outer=fine detail", fontsize=11)
ax[1].plot(cx, cy, "c+", ms=12)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("FIG 2   The SAME image, two descriptions.  Left = what you see.  Right = 'which stripes it is made of'.", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"{OUT}/02_image_and_spectrum.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- FIG 3: low-pass = diffraction
th = np.arctan2(yy - cy, xx - cx)
rr0 = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
star = 0.5 + 0.5 * np.cos(36 * th)         # Siemens star: spokes finer toward center
star[rr0 > 120] = 0.0
radii = [120, 45, 20]
fig, ax = plt.subplots(2, 4, figsize=(15, 7.6))
# col0 original
ax[0, 0].imshow(logmag(spectrum(star)), cmap="magma"); ax[0, 0].plot(cx, cy, "c+", ms=8)
ax[0, 0].set_title("full frequency map", fontsize=10)
ax[1, 0].imshow(star, cmap="gray"); ax[1, 0].set_title("ORIGINAL (sharp)", fontsize=11)
for k, R in enumerate(radii, start=1):
    out, Fm, mask = lowpass(star, R)
    ax[0, k].imshow(logmag(spectrum(star)) * mask, cmap="magma")
    ax[0, k].add_patch(Circle((cx, cy), R, fill=False, color="cyan", lw=2))
    ax[0, k].plot(cx, cy, "c+", ms=8)
    ax[0, k].set_title(f"keep inside circle  R={R}", fontsize=10)
    ax[1, k].imshow(out, cmap="gray")
    ax[1, k].set_title("what the lens sees", fontsize=11)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([]); a.set_xlim(0, N); a.set_ylim(N, 0)
fig.suptitle("FIG 3   The lens keeps ONLY what is inside the circle (cyan).  Throw away outside = blur.\n"
             "Smaller circle = more fine detail thrown away = blurrier.  THIS is the diffraction limit.", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(f"{OUT}/03_lowpass_diffraction.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- FIG 4: two points -> resolution
def two_points(sep):
    img = np.zeros((N, N))
    for s in (-sep/2, sep/2):
        img += np.exp(-(((xx - (cx + s)) ** 2 + (yy - cy) ** 2)) / (2 * 1.2 ** 2))
    return img
pts = two_points(16)
big = lowpass_soft(pts, 20)   # high resolution (sharp): clearly two
small = lowpass_soft(pts, 5)  # low resolution (blurry): merged into one
fig = plt.figure(figsize=(13, 6.5))
titles = ["TRUTH: two separate points", "Large circle (high res)\n-> still see TWO", "Small circle (low res)\n-> merged into ONE"]
imgs = [pts, big, small]
for j, (t, im) in enumerate(zip(titles, imgs)):
    axim = fig.add_subplot(2, 3, j + 1); axim.imshow(norm(im), cmap="gray"); axim.set_title(t, fontsize=11)
    axim.set_xticks([]); axim.set_yticks([])
    axpr = fig.add_subplot(2, 3, j + 4)
    prof = im[int(cy), :]; axpr.plot(prof, color="k"); axpr.set_xlim(cx-30, cx+30)
    axpr.set_title("brightness along the line", fontsize=9); axpr.set_yticks([])
fig.suptitle("FIG 4   'Resolution' = the smallest gap you can still see as TWO things.\n"
             "Bigger circle (radius) -> can separate closer points -> higher resolution (smaller nm).", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(f"{OUT}/04_two_points_resolution.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- FIG 5: SIM enlarges the circle
R0 = 18
wf, _, _ = lowpass(star, R0)        # widefield: small circle
sim, _, _ = lowpass(star, 2 * R0)   # SIM: doubled circle
fig, ax = plt.subplots(1, 3, figsize=(14, 5))
ax[0].imshow(star, cmap="gray"); ax[0].set_title("True structure", fontsize=11)
ax[1].imshow(wf, cmap="gray"); ax[1].set_title(f"Wide-field: circle R={R0}\n(blurry, center lost)", fontsize=11)
ax[2].imshow(sim, cmap="gray"); ax[2].set_title(f"SIM: circle R={2*R0} (2x)\n(detail recovered)", fontsize=11)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("FIG 5   SIM does NOT improve the lens. It uses stripes+math to enlarge the usable circle (~2x).\n"
             "Bigger circle = finer stripes allowed back in = sharper image.", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(f"{OUT}/05_sim_enlarge_circle.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- FIG 6: moire = carrying fine info in
f = 30
ang = np.deg2rad(10)
fine  = 0.5 + 0.5 * np.cos(2 * np.pi * f * (xx - cx) / N)             # unknown fine detail
illum = 0.5 + 0.5 * np.cos(2 * np.pi * f * ((xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)) / N)  # known stripe, slightly rotated
prod  = fine * illum
fig, ax = plt.subplots(1, 3, figsize=(14, 5))
ax[0].imshow(fine, cmap="gray"); ax[0].set_title("Fine detail (too fine,\nnear/over the limit)", fontsize=11)
ax[1].imshow(illum, cmap="gray"); ax[1].set_title("x  Known stripe illumination\n(slightly rotated)", fontsize=11)
ax[2].imshow(prod, cmap="gray"); ax[2].set_title("=  MOIRE: big coarse bands appear\n(carry the fine info, now VISIBLE)", fontsize=11)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("FIG 6   Moire effect: fine (invisible) detail x known stripe = coarse (visible) bands.\n"
             "The stripe 'carries' out-of-circle high-freq info INTO the visible circle. Math then carries it back.", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(f"{OUT}/06_moire.png", dpi=110); plt.close(fig)

print("done; wrote 6 figures to", OUT)
