# Optical DPU Super-Resolution

All-optical **diffractive neural network** (Diffractive Processing Unit, DPU) for
microscopy image super-resolution, simulated end-to-end in PyTorch. The network
restores wide-field, low-SNR microscopy images toward SIM-quality ground truth
using **only physically-realizable optical operations** — no electronic neural
network is attached.

## What makes it "all-optical"

This is a faithful optical-computing model, not a CNN with an optics flavour:

- **Only trainable parameters = phase masks φ** — per-pixel thin-element phase
  modulation, `U · exp(iφ)`.
- **Only nonlinearity = square-law detection `|U|²`** at the detector — there is
  no electronic activation function anywhere.
- **The output is read directly off the detector** (`|U|²`); nothing trainable
  follows the optics.

Free-space propagation between layers is the band-limited **angular spectrum
method** (`H = exp(i·2π·z·√(1/λ² − fx² − fy²))`), with λ = 532 nm and pixel
pitch ≥ 2λ (the SLM Nyquist condition). The task is framed as **same-grid**
resolution enhancement (deconvolution + denoising): the SIM ground truth is
downsampled onto the wide-field grid so the optical `|U|²` output and the target
share the same 128×128 sampling.

## Repository layout

| Path | Contents |
|------|----------|
| `code/optics/` | Core physics & model: `propagation.py` (angular-spectrum diffraction + `|U|²` detection), `dpu.py` (DPU block / stack), `dpu_resnet.py` (coherent-residual DPU-ResNet) |
| `code/datasets/` | BioSR loader + raw `.mrc` → 128×128 WF/GT preprocessing |
| `code/metrics/` | PSNR / SSIM / MS-SSIM with per-image linear rescaling |
| `code/experiments/` | training, SNR-stratified evaluation, figure generation, MNIST optical-classification sanity check |
| `code/baselines/` | DFCAN (electronic reference network) |
| `code/tests/` | physics & gradient regression tests (angular spectrum, DPU, ResNet) |
| `*_demo/` | small standalone scripts that visualise the underlying concepts |

## Setup

```bash
pip install -r requirements.txt   # install torch matching your CUDA/CPU from pytorch.org
```

## Data & artifacts are NOT included

This repository is **code only**:

- The **BioSR** dataset (Qiao et al., *Nature Methods* 2021) is **not** redistributed
  here — download it from the authors' figshare and point
  `BIOSR_ROOT` in `code/datasets/biosr.py` at your copy.

> File paths inside the scripts are absolute (they were run on a SLURM cluster);
> adjust them for your machine.

## Typical workflow

```bash
# 1. preprocess raw BioSR .mrc -> 128x128 WF/GT pairs
python code/datasets/preprocess_biosr.py --specimen Microtubules --src <raw> --dst <out>

# 2. train the optical DPU-ResNet (only φ is trainable)
python code/experiments/sr_train.py --model dpuresnet --z 15e-6 --blocks 6 --shortcut sharp

# 3. evaluate, stratified by SNR level, with paired bootstrap CIs
python code/experiments/eval_stats.py

# 4. (optional) electronic DFCAN baseline for comparison
python code/baselines/dfcan_train.py
```

## References

- Lin et al., *All-optical machine learning using diffractive deep neural networks*, **Science** 2018 (D²NN).
- Zhou et al., *Large-scale neuromorphic optoelectronic computing with a reconfigurable diffractive processing unit*, **Nature Photonics** 2021 (DPU).
- Qiao et al., *Evaluation and development of deep neural networks for image super-resolution in optical microscopy*, **Nature Methods** 2021 (BioSR / DFCAN).

## License

Released under the [MIT License](LICENSE).
