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
mkdir -p logs/hyperparameter_logs
LOG_DIR="logs/hyperparameter_logs/"
exec 1> >(tee "${LOG_DIR}/hyper_search_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/hyper_search_${SLURM_JOB_ID}.err" >&2)

echo "[GPU] GPU Information:"
nvidia-smi || echo "[WARNING] nvidia-smi command failed, but continuing"
export CUDA_LAUNCH_BLOCKING=1

# ——— Setup virtualenv and install dependencies ———
VENV=~/venvs/mensa
echo "[VENV] Setting up virtual environment..."

# Create venv if it doesn't exist
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi

# Activate venv
source "$VENV/bin/activate"

# Install all dependencies (will skip if already installed)
#echo "[DEPS] Installing dependencies..."
#pip install --upgrade --quiet pip
#pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install --quiet transformers diffusers accelerate peft datasets
#pip install --quiet 'numpy<2' scipy pandas scikit-learn matplotlib seaborn
#pip install --quiet optuna plotly joblib kaleido Pillow tqdm

echo "[DEPS] Setup complete!"

cd /work/dlclarge2/ceylanb-DL_Lab_Project/mensa-lora

echo "[HYPERSEARCH] Starting hyperparameter search..."
python3 src/hyper_search.py \
    --dataset_csv ./dataset/dataset.csv \
    --epochs 25 \
    --batch_size 6 \
    --n_trials 40 \

echo "[FINISH] $(date)"
echo "Hyperparameter search complete"
echo "============================================================"
