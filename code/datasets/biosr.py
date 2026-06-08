"""
BioSR 微管数据集 loader（DeepTrackAI 预处理子集）。

事实(已核实)：WF(低SNR低分辨) 与 GT-SIM(高分辨) 均为 128×128 float32，~[0,1]，
**同一网格**——任务是同网格分辨率增强(去卷积+去噪)，非 2× 上采样。
训练/验证：同名文件配对。测试：9 个 SNR level 的 WF 共享同一 GT（用于按信噪比分层评估）。
真实低 SNR 数据自带噪声，无需合成噪声。
"""

import os
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

BIOSR_ROOT = "/fs04/scratch2/mz32/wuyujun/data/BioSR/Microtubules"


def prctile_norm(x, pmin=0.1, pmax=99.9):
    """Qiao 2021 Eq.12 百分位归一化到 [0,1]。x: numpy 数组。"""
    lo = np.percentile(x, pmin)
    hi = np.percentile(x, pmax)
    return np.clip((x - lo) / (hi - lo + 1e-7), 0.0, 1.0).astype(np.float32)


def _load(path):
    a = tifffile.imread(path).astype(np.float32)
    if a.ndim == 3:                      # [H,W,1] -> [H,W]
        a = a[..., 0]
    return a


class BioSRPairs(Dataset):
    """train / validate：同名 WF↔GT 配对。返回 (wf[1,H,W], gt[1,H,W])。"""

    def __init__(self, split="training", root=BIOSR_ROOT, normalize=True):
        assert split in ("training", "validate")
        self.wf_dir = os.path.join(root, f"{split}_wf")
        self.gt_dir = os.path.join(root, f"{split}_gt")
        gt_set = set(os.listdir(self.gt_dir))
        self.files = [f for f in sorted(os.listdir(self.wf_dir)) if f in gt_set]
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        wf = _load(os.path.join(self.wf_dir, f))
        gt = _load(os.path.join(self.gt_dir, f))
        if self.normalize:
            wf, gt = prctile_norm(wf), prctile_norm(gt)
        return torch.from_numpy(wf)[None], torch.from_numpy(gt)[None]   # [1,H,W]


class BioSRTestLevel(Dataset):
    """test：指定 SNR level 的 WF 与共享 GT 配对（按信噪比分层评估用）。"""

    def __init__(self, level="level_01", root=BIOSR_ROOT, normalize=True):
        self.wf_dir = os.path.join(root, "test_wf", level)
        self.gt_dir = os.path.join(root, "test_gt")
        gt_set = set(os.listdir(self.gt_dir))
        self.files = [f for f in sorted(os.listdir(self.wf_dir)) if f in gt_set]
        self.level = level
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        wf = _load(os.path.join(self.wf_dir, f))
        gt = _load(os.path.join(self.gt_dir, f))
        if self.normalize:
            wf, gt = prctile_norm(wf), prctile_norm(gt)
        return torch.from_numpy(wf)[None], torch.from_numpy(gt)[None]


def list_test_levels(root=BIOSR_ROOT):
    return sorted(os.listdir(os.path.join(root, "test_wf")))
