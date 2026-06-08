#!/bin/bash
#SBATCH --job-name=wuyujun_sr
#SBATCH --partition=gpu
#SBATCH --account=mz32
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/fs04/scratch2/mz32/wuyujun/code/logs/sr_%x_%j.out
#SBATCH --error=/fs04/scratch2/mz32/wuyujun/code/logs/sr_%x_%j.err

set -e
PY=/scratch2/mz32/bin/conda_envs/wuyujun/bin/python
cd /fs04/scratch2/mz32/wuyujun/code
echo "node=$(hostname)  job=$SLURM_JOB_ID  model=${MODEL:-dpustack}  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

LINEAR_FLAG=""; [ "${LINEAR:-0}" = "1" ] && LINEAR_FLAG="--linear"
$PY experiments/sr_train.py \
    --model "${MODEL:-dpustack}" --blocks "${BLOCKS:-6}" --n_phase "${NPHASE:-1}" \
    --z "${Z:-300e-6}" --epochs "${EPOCHS:-25}" --batch 32 --lr "${LR:-0.02}" \
    --ssim_w "${SSIMW:-0.1}" --seed "${SEED:-0}" --shortcut "${SHORTCUT:-sharp}" \
    --data_root "${DATA_ROOT:-/fs04/scratch2/mz32/wuyujun/data/BioSR/Microtubules}" \
    --tag "${TAG:-}" $LINEAR_FLAG --device cuda

echo "end=$(date)"
