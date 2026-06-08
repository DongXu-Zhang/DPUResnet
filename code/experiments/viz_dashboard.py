"""
结果总览仪表盘（中文版）：一图看清 F-actin/ER 的 DPU 性能提升状态。
4 面板：(A)改进幅度+电子对照 (B)F-actin SNR趋势翻转 (C)深度规律(结构相关) (D)回收率前后。
数据均来自已验证的 eval_stats / 训练日志(带 95% CI)。输出 results/fig_dashboard.png。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# ---- 中文字体(Noto Sans SC: Latin+希腊+中文俱全)----
_FB = os.path.expanduser("~/.fonts/NotoSansSC.ttf")
fm.fontManager.addfont(_FB)
plt.rcParams["font.family"] = fm.FontProperties(fname=_FB).get_name()
plt.rcParams["axes.unicode_minus"] = False

R = "/fs04/scratch2/mz32/wuyujun/results"

# ---- 已验证数据(eval_stats / 日志)----
snr_mt = list(range(1, 10))
opt_mt = [1.69, 2.55, 2.92, 3.18, 3.34, 3.41, 3.44, 3.54, 3.59]
fac_lv = list(range(1, 13))
opt_fac_12k = [1.69, 1.31, 1.11, 0.95, 0.88, 0.73, 0.68, 0.67, 0.66]
opt_fac_40k = [0.73, 1.39, 1.60, 1.79, 1.86, 1.95, 2.02, 2.07, 2.08, 2.12, 2.15, 2.17]
fac_depth_x = [6, 12, 20]
fac_depth_y = [2.21, 1.90, 1.24]
er_depth_x = [6, 10, 12, 14, 20]
er_depth_y = [2.45, 2.81, 2.80, 2.71, 2.51]
bar = {
    "F-actin 第9层": {"old": [0.66, 0.57, 0.72], "new": [2.08, 1.92, 2.25], "dfcan": [4.65, 4.15, 5.15]},
    "ER 第6层":      {"old": [1.73, 1.58, 1.88], "new": [2.77, 2.62, 2.93], "dfcan": [6.48, 6.10, 6.87]},
}
recov = {"微管 MT": (61, 61), "F-actin": (14, 45), "ER": (27, 43)}

def yerr(t):
    m, lo, hi = t
    return np.array([[m - lo], [hi - m]])

fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# A
a = ax[0, 0]
groups = list(bar.keys()); x = np.arange(len(groups)); w = 0.25
for k, (key, color, lab) in enumerate([("old", "#bbbbbb", "光学 旧 (12k 数据)"),
                                       ("new", "#d62728", "光学 新 (40k + 最优深度)"),
                                       ("dfcan", "#1f77b4", "电子 DFCAN (40k)")]):
    vals = [bar[g][key][0] for g in groups]
    errs = np.hstack([yerr(bar[g][key]) for g in groups])
    a.bar(x + (k - 1) * w, vals, w, yerr=errs, capsize=4, color=color, label=lab)
    for i, v in enumerate(vals):
        a.text(x[i] + (k - 1) * w, v + 0.12, f"{v:.2f}", ha="center", fontsize=8)
a.set_xticks(x); a.set_xticklabels(groups); a.set_ylabel("超 floor 净增益 ΔPSNR (dB)")
a.set_title("(A) DPU 改进幅度 与 距电子的差距(95% 置信区间)")
a.legend(fontsize=9, loc="upper left"); a.grid(axis="y", alpha=0.3)

# B
b = ax[0, 1]
b.plot(fac_lv[:9], opt_fac_12k, "o--", color="#bbbbbb", label="光学 旧12k(增益随 SNR 下降 ↓)")
b.plot(fac_lv, opt_fac_40k, "o-", color="#d62728", label="光学 新40k(增益随 SNR 上升 ↑)")
b.plot(snr_mt, opt_mt, "s-", color="#2ca02c", alpha=0.6, label="微管冠军(参照, 上升)")
b.set_xlabel("信噪比等级(低 → 高)"); b.set_ylabel("超 floor 净增益 (dB)")
b.set_title("(B) F-actin:补足数据后 SNR 趋势恢复健康")
b.legend(fontsize=9); b.grid(alpha=0.3)

# C
c = ax[1, 0]
c.plot(fac_depth_x, fac_depth_y, "o-", color="#d62728", label="F-actin @第12层(峰在浅 b6)")
c.plot(er_depth_x, er_depth_y, "s-", color="#1f77b4", label="ER @第6层(峰在深 b10-12)")
c.scatter([6], [2.21], s=160, facecolors="none", edgecolors="#d62728", linewidths=2)
c.scatter([12], [2.80], s=160, facecolors="none", edgecolors="#1f77b4", linewidths=2)
c.set_xlabel("网络深度(block 数)"); c.set_ylabel("超 floor 净增益 (dB)")
c.set_title("(C) 最优深度随结构而变\n(局部丝状 → 浅; 大尺度网状 → 深)")
c.legend(fontsize=9); c.grid(alpha=0.3)

# D
d = ax[1, 1]
structs = list(recov.keys()); x = np.arange(len(structs)); w = 0.35
old = [recov[s][0] for s in structs]; new = [recov[s][1] for s in structs]
d.bar(x - w / 2, old, w, color="#bbbbbb", label="改进前(12k 数据)")
d.bar(x + w / 2, new, w, color="#d62728", label="改进后(40k + 最优深度)")
for i in range(len(structs)):
    d.text(x[i] - w / 2, old[i] + 1, f"{old[i]}%", ha="center", fontsize=9)
    d.text(x[i] + w / 2, new[i] + 1, f"{new[i]}%", ha="center", fontsize=9)
d.set_xticks(x); d.set_xticklabels(structs); d.set_ylabel("光学回收电子增益的比例 (%)")
d.set_title("(D) 回收电子增益的比例:改进前 → 后")
d.set_ylim(0, 75); d.legend(fontsize=9); d.grid(axis="y", alpha=0.3)
d.text(0, 64, "微管始终用 41k\n(参照,无前后)", fontsize=8, ha="center", color="gray")

fig.suptitle("全光 DPU 在 F-actin / ER 上的性能提升总览(更多数据 + 逐结构最优深度)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(R, "fig_dashboard.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("[saved]", out)
