#!/bin/bash
#SBATCH --job-name=mensa-lora-hypersearch
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=23:30:00

echo "============================================================"
echo "              MENSA LORA HYPERPARAMETER SEARCH"
echo "============================================================"
echo "[JOB] Job ID: $SLURM_JOB_ID"
echo "[NODE] Node: $(hostname)"
echo "[START] $(date)"

# Create logs directory
mkdir -p logs
LOG_FILE="logs/hypersearch_${SLURM_JOB_ID}.out"
exec 1> >(tee "$LOG_FILE")
exec 2> >(tee "$LOG_FILE" >&2)

echo "[GPU] GPU Information:"
nvidia-smi || echo "[WARNING] nvidia-smi command failed, but continuing"

echo "[CUDA] Setting up CUDA environment..."
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"


echo "[DEPS] Installing Optuna for hyperparameter optimization..."
pip install --user --quiet optuna

cd /work/dlclarge2/ceylanb-DL_Lab_Project/mensa-lora

echo "[HYPERSEARCH] Starting hyperparameter search..."
python3 hyper_search.py \
    --dataset_csv ./dataset/dataset.csv \
    --epochs 50 \
    --batch_size 6 \
    --n_trials 30

echo "[FINISH] $(date)"
echo "Hyperparameter search complete"
echo "============================================================"
