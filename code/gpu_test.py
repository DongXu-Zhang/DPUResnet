#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# wuyujun 项目 —— GPU 可用性 & 计算正确性自检脚本
# ----------------------------------------------------------------------------
# 用途:确认 PyTorch 能看到 GPU、能在 GPU 上正确计算、cuDNN/autograd 正常。
# 用法(需在 GPU 计算节点上跑,登录节点没有 GPU):
#     module load pytorch/2.7.0
#     python code/gpu_test.py
#   或直接交互式占一块卡:
#     srun --partition=gpu --account=mz32 --qos=normal \
#          --gres=gpu:L40S:1 --cpus-per-task=4 --mem=16G --time=00:15:00 \
#          --pty bash -c "module load pytorch/2.7.0 && python code/gpu_test.py"
#
# 退出码:0 = 全部通过;1 = 没有可用 GPU;2 = 计算结果不正确/出错。
# ============================================================================
import sys
import time

import torch


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    section("1. 环境信息")
    print(f"  PyTorch 版本        : {torch.__version__}")
    print(f"  编译所用 CUDA 版本   : {torch.version.cuda}")
    cudnn_ver = torch.backends.cudnn.version()
    print(f"  cuDNN 版本          : {cudnn_ver}")

    # ---- 2. GPU 是否可见 ---------------------------------------------------
    section("2. GPU 可用性")
    if not torch.cuda.is_available():
        print("  ✗ torch.cuda.is_available() = False —— 当前没有可用的 GPU。")
        print("    · 如果你在登录节点:登录节点没有 GPU,请通过 SLURM 申请计算节点。")
        print("    · 如果你在 GPU 节点:检查 --gres=gpu:... 是否申请成功、")
        print("      驱动/CUDA 是否匹配(nvidia-smi 是否正常)。")
        return 1

    n = torch.cuda.device_count()
    print(f"  ✓ torch.cuda.is_available() = True,可见 {n} 块 GPU:")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        mem_gb = p.total_memory / 1024 ** 3
        print(f"    [{i}] {p.name} | 算力 sm_{p.major}{p.minor} | "
              f"显存 {mem_gb:.1f} GB | SM 数 {p.multi_processor_count}")

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    print(f"  当前使用设备        : cuda:0 ({torch.cuda.get_device_name(0)})")

    # ---- 3. 计算正确性:GPU 矩阵乘法 vs CPU --------------------------------
    section("3. 计算正确性 (GPU matmul 与 CPU 对拍)")
    torch.manual_seed(0)
    a_cpu = torch.randn(1024, 1024, dtype=torch.float32)
    b_cpu = torch.randn(1024, 1024, dtype=torch.float32)
    ref = a_cpu @ b_cpu                       # CPU 参考结果

    a_gpu, b_gpu = a_cpu.to(dev), b_cpu.to(dev)
    out = (a_gpu @ b_gpu).cpu()                # GPU 计算后取回 CPU

    max_err = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
    print(f"  与 CPU 结果最大绝对误差: {max_err:.3e}")
    if ok:
        print("  ✓ GPU 计算结果与 CPU 一致 (allclose 通过)")
    else:
        print("  ✗ GPU 计算结果与 CPU 不一致 —— 显卡/驱动可能有问题!")
        return 2

    # ---- 4. autograd + cuDNN 反向传播测试 ---------------------------------
    section("4. autograd / cuDNN 反向传播")
    try:
        x = torch.randn(64, 3, 32, 32, device=dev, requires_grad=True)
        conv = torch.nn.Conv2d(3, 16, 3, padding=1).to(dev)
        y = torch.relu(conv(x)).mean()
        y.backward()
        gnorm = x.grad.norm().item()
        print(f"  前向输出 = {y.item():.6f} | 输入梯度范数 = {gnorm:.6f}")
        if x.grad is not None and gnorm > 0:
            print("  ✓ 卷积前向 + 反向(cuDNN/autograd)正常")
        else:
            print("  ✗ 反向传播未得到有效梯度")
            return 2
    except Exception as e:
        print(f"  ✗ 反向传播出错: {e!r}")
        return 2

    # ---- 5. 性能粗测:大矩阵乘法 TFLOPS -----------------------------------
    section("5. 性能粗测 (fp32 大矩阵乘法)")
    size = 8192
    x = torch.randn(size, size, device=dev, dtype=torch.float32)
    y = torch.randn(size, size, device=dev, dtype=torch.float32)
    # 预热,触发 cuBLAS 初始化与 kernel 编译
    for _ in range(3):
        _ = x @ y
    torch.cuda.synchronize()

    iters = 10
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = x @ y
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters          # 每次耗时 (毫秒)
    flops = 2.0 * size ** 3                        # 一次 NxN 矩乘的浮点运算数
    tflops = flops / (ms / 1e3) / 1e12
    print(f"  {size}x{size} fp32 matmul: {ms:.2f} ms/次 ≈ {tflops:.1f} TFLOP/s")

    # ---- 6. 显存分配/回收 -------------------------------------------------
    section("6. 显存分配 / 回收")
    alloc = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"  已分配 {alloc:.0f} MB | 已预留 {reserved:.0f} MB | 峰值 {peak:.0f} MB")
    del x, y
    torch.cuda.empty_cache()
    print(f"  empty_cache 后已分配: {torch.cuda.memory_allocated() / 1024 ** 2:.0f} MB")

    section("结论")
    print("  ✓✓✓ GPU 可用,且计算结果正确 —— 全部测试通过。")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:  # noqa: BLE001
        print(f"\n[未捕获异常] {type(e).__name__}: {e}")
        code = 2
    sys.exit(code)
