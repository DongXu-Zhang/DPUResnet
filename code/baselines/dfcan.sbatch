#!/bin/bash
#SBATCH --job-name=dfcan
#SBATCH --partition=gpu
#SBATCH --account=mz32
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/fs04/scratch2/mz32/wuyujun/code/logs/sr_%x_%j.out
#SBATCH --error=/fs04/scratch2/mz32/wuyujun/code/logs/sr_%x_%j.err

set -e
PY=/scratch2/mz32/bin/conda_envs/wuyujun/bin/python
cd /fs04/scratch2/mz32/wuyujun/code
echo "node=$(hostname)  job=$SLURM_JOB_ID  DFCAN baseline  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 防显存碎片 OOM(尤其 T4 15G)

$PY baselines/dfcan_train.py \
    --channel "${CHANNEL:-64}" --n_group "${NGROUP:-4}" --n_fcab "${NFCAB:-4}" \
    --epochs "${EPOCHS:-80}" --batch "${BATCH:-32}" --lr "${LR:-1e-3}" \
    --data_root "${DATA_ROOT:-/fs04/scratch2/mz32/wuyujun/data/BioSR/Microtubules}" \
    --seed "${SEED:-1}" --tag "${TAG:-dfcan}" --device cuda

echo "end=$(date)"
