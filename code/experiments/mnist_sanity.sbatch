#!/bin/bash
#SBATCH --job-name=wuyujun_mnist
#SBATCH --partition=gpu
#SBATCH --account=mz32
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/fs04/scratch2/mz32/wuyujun/code/logs/mnist_%j.out
#SBATCH --error=/fs04/scratch2/mz32/wuyujun/code/logs/mnist_%j.err

set -e
PY=/scratch2/mz32/bin/conda_envs/wuyujun/bin/python
cd /fs04/scratch2/mz32/wuyujun/code
echo "=========================================="
echo "node=$(hostname)  job=$SLURM_JOB_ID  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

# 安全门2：MNIST 光学分类（DPU 带 |U|² 非线性）
$PY experiments/mnist_sanity.py \
    --grid 128 --blocks 5 --z 300e-6 \
    --epochs 20 --batch 128 --lr 0.03 --nonlinear 1 --device cuda

echo "end=$(date)"
