"""
Demonstrate that a PHASE-ONLY optical network trains like a neural net:
forward = differentiable optical model, backward = autograd computes dL/dphi,
update = Adam on phi only. We OVERFIT one real WF->GT pair to show the mechanism
clearly (this is a teaching demo, not the generalizing model).

Also draws a schematic of the encode -> forward -> loss -> backprop -> update loop.
"""
import os, numpy as np, tifffile, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/fs04/scratch2/mz32/wuyujun/dpu_demo"; os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); torch.set_num_threads(8)
WL, DX, N, NP, Z, L = 698e-9, 9.2e-6, 128, 256, 0.02, 3   # 3 phase layers, 2 cm hops
PI = np.pi
STEPS = 300

def load(p, lo=1, hi=99.5):
    a = tifffile.imread(p).astype(np.float32)
    if a.ndim == 3: a = a[..., 0]
    a = a[:N, :N]
    p1, p2 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - p1) / (p2 - p1 + 1e-7), 0, 1)

R = "/fs04/scratch2/mz32/wuyujun/data/BioSR/Microtubules"
wf = torch.tensor(load(f"{R}/test_wf/level_03/003.tif"))     # blurry, low-SNR input
gt = torch.tensor(load(f"{R}/test_gt/003.tif"))              # sharp target

# ---- angular-spectrum transfer function (precomputed) ----
fx = torch.fft.fftfreq(NP, d=DX)
FX, FY = torch.meshgrid(fx, fx, indexing="ij")
arg = 1.0 - (WL*FX)**2 - (WL*FY)**2
H = torch.where(arg >= 0,
                torch.exp(1j*2*PI/WL*Z*torch.sqrt(torch.clamp(arg, min=0.0))),
                torch.zeros((), dtype=torch.complex64)).to(torch.complex64)

def asm(U):  # P_z : free-space diffraction
    return torch.fft.ifft2(torch.fft.fft2(U) * H)

def pad(x):
    o = (NP - N)//2
    out = torch.zeros((NP, NP), dtype=x.dtype)
    out[o:o+N, o:o+N] = x; return out
def crop(x):
    o = (NP - N)//2; return x[o:o+N, o:o+N]
def nrm(x):  # per-image min-max to [0,1] (display only)
    return (x - x.min()) / (x.max() - x.min() + 1e-7)
def align(pred, target):  # smooth affine align a*pred+b (Qiao Eq.14-15), differentiable, NO trainable params
    p, t = pred.flatten(), target.flatten()
    pm, tm = p.mean(), t.mean()
    a = ((p-pm)*(t-tm)).mean() / (((p-pm)**2).mean() + 1e-8)
    return a*pred + (tm - a*pm)

# ---- the ONLY trainable parameter: phase masks (theta -> phi = 2*pi*sigmoid(theta)) ----
theta = (0.05*torch.randn(L, N, N)).requires_grad_(True)   # phase-only weights (small random init)

def forward(input_img):
    U = pad(torch.sqrt(input_img + 1e-6).to(torch.complex64))   # ENCODE: image -> light amplitude
    for k in range(L):
        phi = 2*PI*torch.sigmoid(theta[k])                      # DM: phase modulation (the weights)
        U = U * pad(torch.exp(1j*phi))
        U = asm(U)                                              # DC+OS: diffraction = matmul
        I = crop(U.abs()**2)                                   # CA: |.|^2 detection (nonlinearity)
        U = pad(torch.sqrt(I + 1e-6).to(torch.complex64)) if k < L-1 else None
        if k == L-1: out = I
    return out                                                  # DECODE: final intensity = the image

opt = torch.optim.Adam([theta], lr=0.02)
losses = []
with torch.no_grad():
    out0 = nrm(align(forward(wf), gt).clamp(0, 1)).clone()     # output BEFORE training (display)
for step in range(STEPS):
    opt.zero_grad()
    out = align(forward(wf), gt)                               # scale/offset align (no trainable params)
    loss = ((out - gt)**2).mean()                              # MSE vs GT (forward graph is differentiable)
    loss.backward()                                            # AUTOGRAD: dL/dphi, no optics needed
    opt.step()                                                 # update phi only
    losses.append(loss.item())
with torch.no_grad():
    outF = nrm(align(forward(wf), gt).clamp(0, 1))

def psnr(a, b): return 10*np.log10(1.0/(((a-b)**2).mean().item()+1e-12))
print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} | PSNR(out,GT) {psnr(out0,gt):.2f} -> {psnr(outF,gt):.2f} dB")

# ============================ FIG: real training result
fig = plt.figure(figsize=(15, 4.4))
axL = fig.add_subplot(1, 5, 1)
axL.plot(losses, color="crimson"); axL.set_title("training loss (MSE vs GT)\nonly phi is updated", fontsize=10)
axL.set_xlabel("Adam step"); axL.set_ylabel("loss"); axL.grid(alpha=0.3)
for j, (im, t) in enumerate([(wf, "INPUT  WF\n(blurry)"),
                             (out0, "optical output\n@ step 0 (random phi)"),
                             (outF, f"optical output\n@ step {STEPS} (trained phi)"),
                             (gt, "TARGET  GT\n(sharp)")]):
    a = fig.add_subplot(1, 5, j+2)
    a.imshow(im.detach().numpy(), cmap="gray", vmin=0, vmax=1); a.set_title(t, fontsize=10)
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("PURE phi-ONLY OPTICAL NET, trained by ordinary backprop (one image overfit, to show the mechanism).  "
             "Forward = differentiable optics; backward = autograd dL/dphi; update = Adam on phi.", fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.9]); fig.savefig(f"{OUT}/5_real_phi_training.png", dpi=110); plt.close(fig)

# ============================ FIG: schematic of the loop
fig, ax = plt.subplots(figsize=(15, 5.5)); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 6)
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=fc, ec="black", lw=1.3))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9)
def arrow(x1, y1, x2, y2, c="black", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16, color=c, lw=2))
# forward chain (top, left->right)
box(0.2, 4.2, 2.0, 1.1, "WF input\n(image)", "#dfe7f5")
box(2.6, 4.2, 2.2, 1.1, "ENCODE\nimage -> light\namplitude", "#cfe9d8")
box(5.2, 4.2, 2.5, 1.1, "LAYER 1\n x e^{i phi_1}\n-> diffract P_z\n-> |.|^2", "#f3e2c0")
box(8.1, 4.2, 2.5, 1.1, "LAYER 2\n x e^{i phi_2}\n-> diffract P_z\n-> |.|^2", "#f3e2c0")
box(11.0, 4.2, 1.9, 1.1, "DECODE\noutput image", "#cfe9d8")
box(13.2, 4.2, 1.6, 1.1, "LOSS\nvs GT", "#f5d0d0")
for x in [2.2, 4.8, 7.7, 10.6, 12.9]: arrow(x, 4.75, x+0.4, 4.75)
ax.text(7.5, 5.55, "FORWARD  (runs as a differentiable simulation on the computer; mirrors the real optics)",
        ha="center", fontsize=10, weight="bold", color="#234")
# backward (bottom, right->left)
box(5.0, 1.4, 5.0, 1.0, "AUTOGRAD computes  dLoss/d phi  through FFT, x e^{i phi}, |.|^2\n(no optics in the loop — pure calculus)", "#e7dbf2")
box(11.0, 1.4, 3.6, 1.0, "Adam updates ONLY phi_1, phi_2\nphi = 2*pi*sigmoid(theta)", "#e7dbf2")
arrow(13.9, 4.1, 13.9, 2.45, c="purple")           # loss -> backward
arrow(10.0, 1.9, 10.5, 1.9, c="purple")            # autograd -> update
arrow(7.5, 2.45, 6.3, 4.1, c="purple", style="-|>")# update -> back into layers
ax.text(7.5, 0.7, "BACKWARD  (right -> left): gradients flow back through the SAME simulated forward graph; only phi changes",
        ha="center", fontsize=10, weight="bold", color="#423")
box(0.2, 1.4, 4.2, 1.0, "After training: write the\nlearned phi onto the real SLM\n(then adaptive-train on hardware)", "#d9d9d9")
arrow(6.0, 1.4, 2.4, 1.4, c="gray")
fig.suptitle("PURE phi-ONLY TRAINING LOOP  —  the 'neural network' is the optics; the 'weights' are phi; backprop is autograd through the optical model", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/4_training_loop_schematic.png", dpi=110); plt.close(fig)
print("done ->", OUT)
