"""Amplitude vs phase input encoding, on a real WF image, with the project's own
optics constants (lambda=532nm, pitch=2*lambda, z=200um)."""
import os, numpy as np, tifffile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/fs04/scratch2/mz32/wuyujun/dpu_demo"
WL, DX, N, Z = 532e-9, 1.064e-6, 256, 400e-6      # project constants (z a bit larger for visible phase->intensity)

def load(p, c=128):
    a = tifffile.imread(p).astype(np.float32)
    if a.ndim == 3: a = a[..., 0]
    a = a[:c, :c]; lo, hi = np.percentile(a,1), np.percentile(a,99.5)
    return np.clip((a-lo)/(hi-lo+1e-7), 0, 1)

def pad(U):
    o=(N-U.shape[0])//2; out=np.zeros((N,N),U.dtype); out[o:o+U.shape[0],o:o+U.shape[0]]=U; return out
def crop(U,s=128):
    o=(N-s)//2; return U[o:o+s,o:o+s]
def asm(U,z):
    fx=np.fft.fftfreq(N,d=DX); FX,FY=np.meshgrid(fx,fx)
    arg=1-(WL*FX)**2-(WL*FY)**2; H=np.zeros_like(arg,complex); m=arg>=0
    H[m]=np.exp(1j*2*np.pi/WL*z*np.sqrt(arg[m])); return np.fft.ifft2(np.fft.fft2(U)*H)

img = load("/fs04/scratch2/mz32/wuyujun/data/BioSR/Microtubules/test_wf/level_05/003.tif")

# amplitude encoding: |U0|^2 = image  (phase flat)
U_amp = pad(np.sqrt(img).astype(complex))
# phase encoding: amplitude uniform, image written into phase in [0, pi]
U_phs = pad(np.exp(1j*(img*np.pi)).astype(complex))

rows = [("AMPLITUDE encoding\nU0 = sqrt(image)", U_amp),
        ("PHASE encoding\nU0 = exp(i*pi*image)", U_phs)]
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for r,(name,U) in enumerate(rows):
    amp = crop(np.abs(U)); ph = crop(np.angle(U)); I0 = crop(np.abs(U)**2)
    Iz = crop(np.abs(asm(U,Z))**2)
    ax[r,0].imshow(amp, cmap="gray", vmin=0, vmax=1.1); ax[r,0].set_title(f"{name}\n\namplitude |U0|", fontsize=9)
    ax[r,1].imshow(ph, cmap="twilight", vmin=-np.pi, vmax=np.pi); ax[r,1].set_title("phase  angle(U0)", fontsize=9)
    ax[r,2].imshow(I0, cmap="inferno", vmin=0, vmax=1.15); ax[r,2].set_title("what a camera sees NOW  |U0|^2", fontsize=9)
    ax[r,3].imshow(Iz, cmap="inferno"); ax[r,3].set_title(f"|U|^2 after {Z*1e6:.0f}um propagation", fontsize=9)
    for c in range(4): ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
fig.suptitle("INPUT ENCODING.  Top: image lives in AMPLITUDE -> camera sees the picture directly.  "
             "Bottom: image lives in PHASE -> camera sees ~UNIFORM (picture hidden!), but propagation turns phase into visible intensity.\n"
             "(lambda=532nm, pitch=1.064um, z as labelled — the project's own optics.)", fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.92]); fig.savefig(f"{OUT}/6_amp_vs_phase_encoding.png", dpi=110); plt.close(fig)
print("done", os.path.exists(f"{OUT}/6_amp_vs_phase_encoding.png"))
