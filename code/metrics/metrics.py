"""
评估指标 + 训练损失（torch 实现，可微）。

含：NRMSE / PSNR / SSIM / MS-SSIM / 线性重标定(Eq.14-15) / evaluate()。
说明：相位-only 衍射网络无损但会重分布能量，输出与 GT 存在全局尺度差，
     故评估前做 MSE-最优线性重标定 alpha*pred+beta，保证公平。
（去相关分析分辨率较复杂，留待后续补充；当前用 NRMSE/PSNR/SSIM/MS-SSIM。）
"""

import torch
import torch.nn.functional as F


def _b1hw(x):
    """统一成 [B,1,H,W]。"""
    if x.dim() == 2:
        return x[None, None]
    if x.dim() == 3:
        return x[:, None] if x.shape[0] != 1 else x[None]  # [B,H,W] or [1,H,W]
    return x


def linear_rescale(pred, target):
    """逐图 MSE-最优 alpha*pred+beta 贴合 target。pred/target: [B,1,H,W]。"""
    B = pred.shape[0]
    out = torch.empty_like(pred)
    for i in range(B):
        p = pred[i].flatten()
        t = target[i].flatten()
        A = torch.stack([p, torch.ones_like(p)], dim=1)        # [N,2]
        sol = torch.linalg.lstsq(A, t[:, None]).solution       # [2,1]
        out[i] = (sol[0, 0] * pred[i] + sol[1, 0])
    return out


def nrmse(pred, target):
    pred, target = _b1hw(pred), _b1hw(target)
    num = torch.sqrt(((pred - target) ** 2).mean(dim=(1, 2, 3)))
    den = target.amax(dim=(1, 2, 3)) - target.amin(dim=(1, 2, 3)) + 1e-7
    return (num / den).mean()


def psnr(pred, target, data_range=1.0):
    pred, target = _b1hw(pred), _b1hw(target)
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    return (10 * torch.log10(data_range ** 2 / (mse + 1e-12))).mean()


def _gauss_window(size, sigma, device, dtype):
    c = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    return w[None, None]                                       # [1,1,size,size]


def _ssim_map(pred, target, data_range, win, sigma):
    w = _gauss_window(win, sigma, pred.device, pred.dtype)
    pad = win // 2
    mu1 = F.conv2d(pred, w, padding=pad)
    mu2 = F.conv2d(target, w, padding=pad)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(pred * pred, w, padding=pad) - mu1s
    s2 = F.conv2d(target * target, w, padding=pad) - mu2s
    s12 = F.conv2d(pred * target, w, padding=pad) - mu12
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    cs = (2 * s12 + C2) / (s1 + s2 + C2)                       # 对比/结构项
    ssim = ((2 * mu12 + C1) / (mu1s + mu2s + C1)) * cs
    return ssim, cs


def ssim(pred, target, data_range=1.0, win=11, sigma=1.5):
    pred, target = _b1hw(pred), _b1hw(target)
    s, _ = _ssim_map(pred, target, data_range, win, sigma)
    return s.mean()


def ms_ssim(pred, target, data_range=1.0, win=11, sigma=1.5):
    """3 尺度 MS-SSIM（适配 128×128：128->64->32，window=11 在 32 仍合法）。"""
    pred, target = _b1hw(pred), _b1hw(target)
    weights = torch.tensor([0.25, 0.35, 0.40], device=pred.device, dtype=pred.dtype)
    mcs = []
    for i in range(3):
        s, cs = _ssim_map(pred, target, data_range, win, sigma)
        if i < 2:
            mcs.append(cs.mean().clamp(min=1e-6))
            pred = F.avg_pool2d(pred, 2)
            target = F.avg_pool2d(target, 2)
        else:
            ssim_last = s.mean().clamp(min=1e-6)
    out = ssim_last ** weights[2]
    for i in range(2):
        out = out * (mcs[i] ** weights[i])
    return out


@torch.no_grad()
def evaluate(pred, target, rescale=True):
    """评估一批：先(可选)线性重标定，再算各指标。返回 dict。"""
    pred, target = _b1hw(pred).float(), _b1hw(target).float()
    if rescale:
        pred = linear_rescale(pred, target).clamp(0, 1)
    return {
        "PSNR": psnr(pred, target).item(),
        "SSIM": ssim(pred, target).item(),
        "MS_SSIM": ms_ssim(pred, target).item(),
        "NRMSE": nrmse(pred, target).item(),
    }


def sr_loss(pred, target, ssim_w=0.1):
    """训练损失 = MSE + ssim_w*(1-SSIM)（Qiao 2021 Eq.8，可微）。pred/target: [B,1,H,W] in [0,1]。"""
    pred, target = _b1hw(pred), _b1hw(target)
    return ((pred - target) ** 2).mean() + ssim_w * (1.0 - ssim(pred, target))
