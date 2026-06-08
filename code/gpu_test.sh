#!/bin/bash
# ============================================================================
# wuyujun 项目 —— GPU 自检任务 (提交到计算节点跑 gpu_test.py)
# 提交:  sbatch code/gpu_test.sh
# 查看:  squeue -u $USER   |   tail -f logs/gputest_*.out
# 取消:  scancel <JOBID>
# ============================================================================
#SBATCH --job-name=gputest
#SBATCH --partition=gpu               # gpu(A100/L40S/A40) | fit(A100/H200) | m3h(H100)
#SBATCH --account=mz32                # gpu用mz32; fit用nd32; m3h两者皆可
#SBATCH --qos=normal                  # gpu->normal; fit->fitq; m3h->m3h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1             # 改型号即可换卡: gpu:A100:1 / gpu:A40:1 ...
#SBATCH --mem=16G
#SBATCH --time=00:15:00               # 自检很快,15 分钟足够
#SBATCH --output=logs/gputest_%x_%j.out
#SBATCH --error=logs/gputest_%x_%j.err

set -e
echo "=========================================="
echo "作业ID : $SLURM_JOB_ID"
echo "节点   : $(hostname)"
echo "分区   : $SLURM_JOB_PARTITION"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
echo "开始   : $(date)"
echo "=========================================="

# ---- 先看硬件层面驱动是否正常 ---------------------------------------------
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

# ---- 激活 PyTorch 环境并运行自检 ------------------------------------------
module load pytorch/2.7.0            # torch 2.7.0+cu126 (CUDA 12.6 / cuDNN 9.5)

cd /fs04/scratch2/mz32/wuyujun
python code/gpu_test.py
status=$?

echo "=========================================="
echo "退出码 : $status   (0=通过, 1=无GPU, 2=计算错误)"
echo "结束   : $(date)"
echo "=========================================="
exit $status
