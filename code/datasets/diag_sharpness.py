"""
诊断:各 BioSR 结构 GT 相对 WF 的"高频锐利度比"(可复现版)。

动机:用于判断某结构的 SR 余量(GT 比 WF 锐多少)。高频占比 = FFT 幅度谱在
归一化半径 0.5~1.0 高频带的能量 / (高频带+低频带)。GT/WF 高频占比之比越大,
说明 GT 相对 WF 越锐、SR 余量越大。

用法:python datasets/diag_sharpness.py        # 跑所有已存在的结构, test level_最高
输出打印到 stdout(建议重定向到 logs/diag_sharpness.txt 留痕)。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np                                            # noqa: E402
import numpy.fft as fft                                       # noqa: E402
from torch.utils.data import Subset, DataLoader               # noqa: E402

from datasets.biosr import BioSRTestLevel, list_test_levels   # noqa: E402

BIOSR = "/fs04/scratch2/mz32/wuyujun/data/BioSR"


def hf_lf_energy(img):
    """返回 (高频带能量, 低频带能量)。img: [H,W]。去均值后取 FFT 功率谱。"""
    P = np.abs(fft.fftshift(fft.fft2(img - img.mean()))) ** 2
    H, W = img.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2) / (min(H, W) / 2)
    hi = (r >= 0.5) & (r <= 1.0)
    lo = (r < 0.5) & (r > 0.02)
    return P[hi].sum(), P[lo].sum()


def hf_frac(img):
    h, l = hf_lf_energy(img)
    return h / (h + l + 1e-9)


def diag_structure(root, n=48):
    """用该结构 test 最高 level 的 n 张图,算 WF/GT 高频占比与其比值。"""
    lvls = list_test_levels(root)
    if not lvls:
        return None
    ds = Subset(BioSRTestLevel(lvls[-1], root=root), range(min(n, len(BioSRTestLevel(lvls[-1], root=root)))))
    ld = DataLoader(ds, batch_size=16)
    rw, rg = [], []
    for wf, gt in ld:
        for i in range(wf.shape[0]):
            rw.append(hf_frac(wf[i, 0].numpy()))
            rg.append(hf_frac(gt[i, 0].numpy()))
    rw, rg = float(np.mean(rw)), float(np.mean(rg))
    return rw, rg, rg / (rw + 1e-9)


def main():
    structs = ["Microtubules", "CCPs", "F-actin", "ER"]
    print(f"{'结构':16s} {'WF高频占比':>10} {'GT高频占比':>10} {'GT/WF锐利度比':>14}")
    for s in structs:
        root = os.path.join(BIOSR, s)
        if not os.path.isdir(os.path.join(root, "test_gt")):
            print(f"{s:16s}  (无数据, 跳过)")
            continue
        r = diag_structure(root)
        if r is None:
            print(f"{s:16s}  (无 test level, 跳过)")
            continue
        rw, rg, ratio = r
        print(f"{s:16s} {rw:10.4f} {rg:10.4f} {ratio:14.2f}")


if __name__ == "__main__":
    main()
