#!/bin/bash
# ============================================================================
# wuyujun 项目 —— CPU 批处理任务模板
# 提交:  sbatch code/cpu_job.sh
# ============================================================================
#SBATCH --job-name=wuyujun_cpu
#SBATCH --partition=comp              # comp(默认,48~128核) | genomics | short(≤30min)
#SBATCH --account=mz32
#SBATCH --qos=normal                  # short分区用 shortq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16            # comp单用户上限250核
#SBATCH --mem=64G                     # 大内存可到 ~1.5T(EPYC节点)
#SBATCH --time=1-00:00:00             # comp上限7天
#SBATCH --output=logs/cpu_%x_%j.out
#SBATCH --error=logs/cpu_%x_%j.err

set -e
echo "=========================================="
echo "作业ID : $SLURM_JOB_ID"
echo "节点   : $(hostname)  ($(nproc) 核可见)"
echo "开始   : $(date)"
echo "=========================================="

# ---- 环境激活 (按需) -------------------------------------------------------
# source /fs04/scratch2/mz32/zhaorui/miniconda3/bin/activate <你的env名>

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- 你的命令 --------------------------------------------------------------
cd /fs04/scratch2/mz32/wuyujun
# python code/process.py --threads $SLURM_CPUS_PER_TASK

echo "结束   : $(date)"
