"""
全光 DPU 超分辨项目 —— 全局物理常数（单一可信来源）

所有数值锁在这里，代码里不许散落魔法数字。对应老师文档的提醒：
 (b) 波长可用 532 nm；SLM 单元尺寸 >= 2 倍波长
 (d) 受显存限制，处理图像尺寸取小一点，如 128x128
"""

# --- 光学物理常数 ---
WAVELENGTH = 532e-9          # m, 绿光（提醒 b）
PIXEL_PITCH = 2 * WAVELENGTH  # m, SLM 单元尺寸下限 = 2*lambda = 1.064 um（提醒 b）

# --- 仿真网格 ---
PATCH = 128                  # 处理 patch 边长（提醒 d）
PAD_FACTOR = 2               # FFT 前 zero-pad 倍数，抑制循环卷积绕回

# --- 数值 ---
EPS = 1e-7                   # sqrt(I+eps) 重编码用，避免 0 点梯度爆炸

# --- DPU 架构默认（均为可扫超参，非物理硬约束）---
Z_DEFAULT = 200e-6           # m, DPU 层间传播距离(遗留默认, 适合分类/大z混合)。
                             #   ⚠️ 超分辨成像必须覆盖到 ~15e-6(见 DESIGN 文档 §6.1 P1):
                             #   大z强衍射打乱空间结构, 成像要小z对齐。SR run 一律 --z 15e-6。
                             #   14.5°衍射锥: 横向扩散 ~ z*tan(14.5°)=0.26*z(极端光线上界)。
N_BLOCKS = 4                 # DPU block 数（网络深度）
N_PHASE_PER_BLOCK = 1        # 每个 block 内、末端探测前的相位层数（1=最简纯DPU级联）


def max_diffraction_half_angle_deg(dx=PIXEL_PITCH, wavelength=WAVELENGTH):
    """pitch>=2*lambda 的物理后果：衍射半角受限。dx=2*lambda 时约 14.5 度。"""
    import math
    sin_theta = wavelength / (2.0 * dx)          # = lambda/(2*dx)
    return math.degrees(math.asin(min(sin_theta, 1.0)))
