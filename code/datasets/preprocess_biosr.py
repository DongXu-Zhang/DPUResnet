"""
把原始 BioSR 单结构 (.mrc) 预处理成 DeepTrackAI 同款 128×128 float32 [0,1] WF→GT 数据，
使现有 datasets/biosr.py 仅换 root 即可复用，用于验证光学 DPU 方法在其他结构上的泛化性。

  - WF(宽场低分辨输入) = 9 帧结构光照明原始 SIM 之和（结构光相加→均匀照明=宽场），再百分位归一化。
  - GT = SIM 重建 (2× 网格) 用 2×2 块均值降采样到 WF 同网格（本项目任务=同网格去卷积+去噪，
        光学网络输出也是同 128 网格 |U|²，故 GT 必须同网格）。
  - 训练/验证：同名 WF↔GT 配对。测试：每个 GT 配 9 个 SNR level 的 WF（同裁剪位置）。
  - 前景掩膜：按 GT patch 平均强度过滤空 patch（对 CCPs 这类稀疏点状结构尤其必要）。

用法：
  python datasets/preprocess_biosr.py --specimen CCPs \
      --src /fs04/scratch2/mz32/wuyujun/data/BioSR_raw/CCPs \
      --dst /fs04/scratch2/mz32/wuyujun/data/BioSR/CCPs \
      --n_train 12000 --n_val 600 --n_test 96
"""

import argparse
import os
import glob
import numpy as np
import mrcfile
import tifffile


def prctile_norm(x, pmin=0.1, pmax=99.9):
    lo, hi = np.percentile(x, pmin), np.percentile(x, pmax)
    return np.clip((x - lo) / (hi - lo + 1e-7), 0.0, 1.0).astype(np.float32)


def read_mrc(path):
    with mrcfile.open(path, permissive=True) as m:
        return np.array(m.data, dtype=np.float32)


def wf_from_raw(raw):
    """9 帧原始 SIM -> 宽场 = 帧求和。raw: [9,H,W] 或 [H,W]。"""
    return raw.sum(0) if raw.ndim == 3 else raw


def down2(gt):
    """SIM-GT (2H,2W) -> (H,W)：精确 2×2 块均值降采样（sr_ratio=2，与 WF 同网格）。"""
    H = (gt.shape[-2] // 2) * 2
    W = (gt.shape[-1] // 2) * 2
    g = gt[:H, :W]
    return g.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))


def cell_paths(src):
    cells = sorted(glob.glob(os.path.join(src, "Cell_*")))
    return cells


def is_er_layout(cell):
    """ER 布局: RawSIMData/ 与 GTSIM/ 子目录, GT 逐 level; 其余结构: cell 根目录 + 单一 SIM_gt.mrc。"""
    return os.path.isdir(os.path.join(cell, "RawSIMData"))


def n_levels_of(cell):
    d = os.path.join(cell, "RawSIMData") if is_er_layout(cell) else cell
    return len(glob.glob(os.path.join(d, "RawSIMData_level_*.mrc")))


def raw_path(cell, lv):
    if is_er_layout(cell):
        return os.path.join(cell, "RawSIMData", f"RawSIMData_level_{lv:02d}.mrc")
    return os.path.join(cell, f"RawSIMData_level_{lv:02d}.mrc")


def gt_path(cell, n_levels):
    """ER: 用最高 level(最干净)的 GTSIM 作单一GT, 与其它结构(单一SIM_gt)口径一致。"""
    if is_er_layout(cell):
        return os.path.join(cell, "GTSIM", f"GTSIM_level_{n_levels:02d}.mrc")
    return os.path.join(cell, "SIM_gt.mrc")


def load_cell_wf(cell, level):
    """WF[H,W] 归一化 = 该 level 的 9 帧原始 SIM 求和。"""
    return prctile_norm(wf_from_raw(read_mrc(raw_path(cell, level))))


def load_cell_gt(cell, n_levels):
    """GT[H,W] 同网格归一化(2×降采样)。"""
    return prctile_norm(down2(read_mrc(gt_path(cell, n_levels))))


def load_cell_wf_gt(cell, level, n_levels):
    """返回该 cell 在指定 SNR level 的 (WF, GT)，尺寸对齐。"""
    wf = load_cell_wf(cell, level)
    gt = load_cell_gt(cell, n_levels)
    h = min(wf.shape[0], gt.shape[0])
    w = min(wf.shape[1], gt.shape[1])
    return wf[:h, :w], gt[:h, :w]


def sample_fg_coords(gt, n, size, rng, fg_ratio=0.5, max_try=60):
    """随机采 n 个前景裁剪左上角坐标：GT patch 均值 > fg_ratio×全图均值 才保留。"""
    H, W = gt.shape
    thr = fg_ratio * float(gt.mean())
    coords = []
    tries = 0
    while len(coords) < n and tries < n * max_try:
        tries += 1
        y = int(rng.integers(0, H - size + 1))
        x = int(rng.integers(0, W - size + 1))
        if gt[y:y + size, x:x + size].mean() > thr:
            coords.append((y, x))
    return coords


def save_tif(path, patch):
    tifffile.imwrite(path, patch.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--levels", type=int, default=0, help="0=自动检测(标准9/ER6)")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=96)
    ap.add_argument("--val_cells", type=int, default=5, help="前 N 个 cell 作验证")
    ap.add_argument("--test_cells", type=int, default=12, help="末 N 个 cell 作测试")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cells = cell_paths(args.src)
    nC = len(cells)
    assert nC > args.val_cells + args.test_cells, f"cell 太少: {nC}"
    val_cells = cells[:args.val_cells]
    test_cells = cells[nC - args.test_cells:]
    train_cells = cells[args.val_cells:nC - args.test_cells]
    print(f"[{args.specimen}] cells={nC}  train={len(train_cells)} val={len(val_cells)} test={len(test_cells)}")

    L = args.levels if args.levels > 0 else n_levels_of(cells[0])
    print(f"  SNR levels = {L}  (布局: {'ER(子目录,逐level GT)' if is_er_layout(cells[0]) else '标准(单一SIM_gt)'})")
    sz = args.size
    for sub in ["training_wf", "training_gt", "validate_wf", "validate_gt", "test_gt"]:
        os.makedirs(os.path.join(args.dst, sub), exist_ok=True)
    for lv in range(1, L + 1):
        os.makedirs(os.path.join(args.dst, "test_wf", f"level_{lv:02d}"), exist_ok=True)

    # ---- 训练 / 验证：每 (cell, level) 采若干前景 patch ----
    for split, cell_list, n_target, wf_dir, gt_dir in [
        ("train", train_cells, args.n_train, "training_wf", "training_gt"),
        ("val", val_cells, args.n_val, "validate_wf", "validate_gt"),
    ]:
        per = max(1, int(np.ceil(n_target / (len(cell_list) * L))))
        cnt = 0
        for ci, cell in enumerate(cell_list):
            for lv in range(1, L + 1):
                if cnt >= n_target:
                    break
                wf, gt = load_cell_wf_gt(cell, lv, L)
                rng = np.random.default_rng(args.seed * 100000 + ci * 100 + lv)
                coords = sample_fg_coords(gt, per, sz, rng)
                for (y, x) in coords:
                    if cnt >= n_target:
                        break
                    name = f"{cnt + 1:08d}.tif"
                    save_tif(os.path.join(args.dst, wf_dir, name), wf[y:y + sz, x:x + sz])
                    save_tif(os.path.join(args.dst, gt_dir, name), gt[y:y + sz, x:x + sz])
                    cnt += 1
        print(f"  {split}: 写出 {cnt} 对 (目标 {n_target})")

    # ---- 测试：每个 GT patch 配 9 个 level 的 WF（同裁剪位置）----
    # 用 GT(与 level 无关) 选前景坐标；WF 取各 level 的 9 帧求和（同坐标裁剪）。
    cnt = 0
    per_test = max(1, int(np.ceil(args.n_test / len(test_cells))))
    for ci, cell in enumerate(test_cells):
        # 用统一 GT(最高level降采样)选坐标
        gt_full = load_cell_gt(cell, L)
        rng = np.random.default_rng(999000 + ci)
        coords = sample_fg_coords(gt_full, per_test, sz, rng)
        # 预读各 level 的 WF 全图
        wf_levels = {lv: load_cell_wf(cell, lv) for lv in range(1, L + 1)}
        for (y, x) in coords:
            if cnt >= args.n_test:
                break
            name = f"{cnt + 1:08d}.tif"
            save_tif(os.path.join(args.dst, "test_gt", name), gt_full[y:y + sz, x:x + sz])
            for lv in range(1, L + 1):
                wfl = wf_levels[lv]
                hh = min(wfl.shape[0], gt_full.shape[0])
                ww = min(wfl.shape[1], gt_full.shape[1])
                yy, xx = min(y, hh - sz), min(x, ww - sz)
                save_tif(os.path.join(args.dst, "test_wf", f"level_{lv:02d}", name),
                         wfl[yy:yy + sz, xx:xx + sz])
            cnt += 1
        if cnt >= args.n_test:
            break
    print(f"  test: 写出 {cnt} 个 GT × {L} levels")


if __name__ == "__main__":
    main()
