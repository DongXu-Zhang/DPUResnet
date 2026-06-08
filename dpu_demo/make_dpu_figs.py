"""
Show, with REAL angular-spectrum propagation on a REAL image, how one DPU layer
turns an input into a 'feature map': input field -> phase mask exp(i*phi) ->
free-space diffraction (P_z) -> |.|^2 detection. Pure CS friendly visuals.

Params are slightly scaled for clean, alias-free visualization (smaller z than
Zhou's 20 cm, larger zero-padded grid). The PHYSICS is identical.
"""
import os, numpy as np, tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/fs04/scratch2/mz32/wuyujun/dpu_demo"
os.makedirs(OUT, exist_ok=True)

WL = 698e-9      # wavelength (m)  -- the 'color/ruler' of the light
DX = 9.2e-6      # pixel pitch (m) -- SLM neuron size
N  = 512         # zero-padded grid (gives ASM validity up to ~6 cm)

def load_img(p, crop=128):
    a = tifffile.imread(p).astype(np.float32)
    if a.ndim == 3: a = a[..., 0]
    h, w = a.shape
    y0, x0 = (h-crop)//2, (w-crop)//2
    a = a[y0:y0+crop, x0:x0+crop]
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a-lo)/(hi-lo+1e-7), 0, 1)

def pad(U):
    out = np.zeros((N, N), dtype=U.dtype)
    s = U.shape[0]; o = (N-s)//2
    out[o:o+s, o:o+s] = U
    return out, o

def crop_center(U, s):
    o = (U.shape[0]-s)//2
    return U[o:o+s, o:o+s]

def asm(U, z):
    """Angular-spectrum free-space propagation by distance z (the P_z operator)."""
    fx = np.fft.fftfreq(N, d=DX); FX, FY = np.meshgrid(fx, fx)
    arg = 1.0 - (WL*FX)**2 - (WL*FY)**2
    H = np.zeros_like(arg, dtype=complex)
    m = arg >= 0
    H[m] = np.exp(1j * 2*np.pi/WL * z * np.sqrt(arg[m]))   # evanescent (arg<0) -> 0
    return np.fft.ifft2(np.fft.fft2(U) * H)

# coordinate grid (m) on the padded plane, centered
c = (np.arange(N) - N//2) * DX
XX, YY = np.meshgrid(c, c)

def phase_flat():   return np.zeros((N, N))
def phase_random():
    rng = np.random.RandomState(0); return rng.uniform(0, 2*np.pi, (N, N))
def phase_lens(f):  return (-np.pi/(WL*f) * (XX**2+YY**2)) % (2*np.pi)
def phase_grating(period_px):
    return (2*np.pi * XX/(period_px*DX)) % (2*np.pi)
def phase_smooth(scale=2.0, seed=1):
    """A smooth, low-frequency 'learned-looking' phase mask (reshapes, doesn't focus to a dot)."""
    rng = np.random.RandomState(seed); base = rng.randn(N, N)
    fx = np.fft.fftfreq(N); FX, FY = np.meshgrid(fx, fx)
    lp = np.exp(-(FX**2+FY**2)/(2*(0.012)**2))
    sm = np.real(np.fft.ifft2(np.fft.fft2(base)*lp))
    sm = (sm-sm.min())/(sm.max()-sm.min())
    return sm*scale*np.pi

img = load_img("/fs04/scratch2/mz32/wuyujun/data/BioSR/Microtubules/test_gt/003.tif", 128)
Upad, off = pad(img.astype(complex))     # input field: amplitude = image, phase = 0

# ============================================================ FIG 1: anatomy of one layer
z1 = 0.02
phi = phase_smooth(2.0)                     # a smooth 'learned-looking' phase mask = the trainable weights
field_after_mask = Upad * np.exp(1j*phi)   # exp(i*phi) modulation
prop = asm(field_after_mask, z1)           # P_z diffraction
feat = np.abs(prop)**2                      # |.|^2 detection  -> feature map

S = 180
fig, ax = plt.subplots(1, 5, figsize=(19, 4.2))
ax[0].imshow(crop_center(np.abs(Upad), S), cmap="gray"); ax[0].set_title("1) INPUT  U^(l)\n(image = light amplitude)", fontsize=10)
ax[1].imshow(crop_center(phi, S), cmap="twilight"); ax[1].set_title("2) PHASE MASK  phi  (the SLM)\n= the trainable WEIGHTS", fontsize=10)
ax[2].imshow(crop_center(np.angle(field_after_mask), S), cmap="twilight"); ax[2].set_title("3) after  U * exp(i*phi)\n(wavefront now reshaped)", fontsize=10)
ax[3].imshow(crop_center(np.abs(prop), S), cmap="gray"); ax[3].set_title("4) after diffraction  P_z\n(light flew z, mixed = matmul)", fontsize=10)
ax[4].imshow(crop_center(feat, S), cmap="inferno"); ax[4].set_title("5) DETECT |.|^2  -> U^(l+1)\n= the FEATURE MAP (nonlinear)", fontsize=10)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("ONE DPU LAYER = one neural-net layer.   out = |  P_z[ U * exp(i*phi) ]  |^2   <=>   out = activation( W · in ).   "
             "Light does W·in for free; phi sets W; the camera's |.|^2 is the activation.", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.9]); fig.savefig(f"{OUT}/1_one_layer_anatomy.png", dpi=110); plt.close(fig)

# ============================================================ FIG 2: phi IS the weight
z2 = 0.03
masks = [("phi = 0  (no weights,\njust free diffraction)", phase_flat()),
         ("phi = random\n(a random projection)", phase_random()),
         ("phi = lens\n(focuses / 'detects a blob')", phase_lens(z2)),
         ("phi = grating\n(shifts / 'edge-steer')", phase_grating(8))]
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for k, (name, phi) in enumerate(masks):
    out = np.abs(asm(Upad*np.exp(1j*phi), z2))**2
    ax[0, k].imshow(crop_center(phi, S), cmap="twilight"); ax[0, k].set_title(f"SLM pattern:\n{name}", fontsize=9)
    ax[1, k].imshow(crop_center(out, S), cmap="inferno"); ax[1, k].set_title("-> resulting feature map", fontsize=9)
    for r in (0, 1): ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
fig.suptitle("SAME input image, SAME distance, DIFFERENT phase mask -> DIFFERENT feature map.   "
             "This is how the SLM 'computes a feature': training just searches for the phi that yields the useful feature.", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(f"{OUT}/2_phi_is_the_weight.png", dpi=110); plt.close(fig)

# ============================================================ FIG 3: distance z = connectivity (fan-out)
pt = np.zeros((128, 128));
yy, xx = np.mgrid[0:128, 0:128]
pt = np.exp(-(((xx-64)**2+(yy-64)**2)/2.0))   # one tiny bright point (Gaussian)
ptpad, _ = pad(pt.astype(complex))
zs = [0.0, 0.01, 0.03, 0.05]
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for k, z in enumerate(zs):
    # top: a single input POINT fans out with distance
    sp = np.abs(asm(ptpad, z))**2 if z > 0 else np.abs(ptpad)**2
    ax[0, k].imshow(crop_center(sp/sp.max(), 220), cmap="inferno");
    ax[0, k].set_title(f"one input point, z = {z*100:.0f} cm", fontsize=10)
    # bottom: the whole image at same z (no mask) -> increasing blur/mix
    im = np.abs(asm(Upad, z))**2 if z > 0 else np.abs(Upad)**2
    ax[1, k].imshow(crop_center(im, S), cmap="gray");
    ax[1, k].set_title(f"image, z = {z*100:.0f} cm", fontsize=10)
    for r in (0, 1): ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
fig.suptitle("DISTANCE z = how far light flies = how widely one input point spreads = HOW MANY output neurons it connects to.   "
             "Longer z -> wider fan-out -> 'more fully connected'. Zhou picks z=20 cm so one point reaches the WHOLE output plane.", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(f"{OUT}/3_distance_is_connectivity.png", dpi=110); plt.close(fig)

print("done ->", OUT)
for f in ("1_one_layer_anatomy.png","2_phi_is_the_weight.png","3_distance_is_connectivity.png"):
    print("  ", f, os.path.exists(f"{OUT}/{f}"))
