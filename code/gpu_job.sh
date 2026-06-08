#!/bin/bash
# ============================================================================
# wuyujun 项目 —— GPU 批处理任务模板
# 提交:  sbatch code/gpu_job.sh
# 查看:  squeue -u $USER   |   tail -f logs/gpu_*.out
# 取消:  scancel <JOBID>
# ============================================================================
#SBATCH --job-name=wuyujun_gpu
#SBATCH --partition=gpu               # gpu(A100/L40S) | fit(A100/H200) | m3h(H100)
#SBATCH --account=mz32                # gpu用mz32; fit用nd32; m3h两者皆可
#SBATCH --qos=normal                  # gpu->normal; fit->fitq; m3h->m3h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16            # 每块GPU配8~16核较合理
#SBATCH --gres=gpu:L40S:1             # 改型号/数量: gpu:A100:1 / gpu:L40S:4 ...
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00             # 上限: gpu/normal=7天, fit=1天, m3h=2天
#SBATCH --output=logs/gpu_%x_%j.out
#SBATCH --error=logs/gpu_%x_%j.err

set -e
echo "=========================================="
echo "作业ID : $SLURM_JOB_ID"
echo "节点   : $(hostname)"
echo "分区   : $SLURM_JOB_PARTITION"
echo "GPU    : $CUDA_VISIBLE_DEVICES"
echo "开始   : $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- 环境激活 (按需取消注释其中一种) --------------------------------------
# source /fs04/scratch2/mz32/zhaorui/miniconda3/bin/activate <你的env名>
# source .venv/bin/activate
# module load cuda

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ---- 你的命令 --------------------------------------------------------------
cd /fs04/scratch2/mz32/wuyujun
# python code/train.py --config code/config.yaml

echo "结束   : $(date)"
