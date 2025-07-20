#!/bin/bash
#SBATCH --job-name=mensa-lora-train
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=02:30:00

echo "============================================================"
echo "              MENSA LORA TRAINING JOB"
echo "============================================================"
echo "[JOB] Job ID: $SLURM_JOB_ID"
echo "[NODE] Node: $(hostname)"

# Configuration
EXPERIMENT_NAME="${1:-experiment_default}"

# Set up experiment-specific log directory
LOG_DIR="logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

# Redirect output to experiment-specific log files
exec 1> >(tee "${LOG_DIR}/train_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/train_${SLURM_JOB_ID}.err" >&2)

echo "[EXP] Experiment: $EXPERIMENT_NAME"
echo "[LOG] Logs directory: ${LOG_DIR}/"
echo "[OUT] Output log: ${LOG_DIR}/train_${SLURM_JOB_ID}.out"
echo "[ERR] Error log: ${LOG_DIR}/train_${SLURM_JOB_ID}.err"

# Create directories
mkdir -p logs

echo "[GPU] GPU Information:"
nvidia-smi

echo "[GPU] Checking GPU stability..."
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

echo "[CUDA] Setting up CUDA environment..."
export CUDA_LAUNCH_BLOCKING=1 # To see CUDA errors, delete for full power
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"

echo "[DEPS] Installing remaining dependencies for LoRA training..."
# Silence pip
export PIP_DISABLE_PIP_VERSION_CHECK=1
pip3 install --user --disable-pip-version-check --quiet peft accelerate datasets \
  > /dev/null 2>&1


echo "[TRAIN] Starting LoRA training..."
cd /work/dlclarge2/ceylanb-DL_Lab_Project/mensa-lora

# Run training
python3 train_lora_enhanced.py \
    --dataset_csv ./dataset/dataset.csv \
    --experiment_name "$EXPERIMENT_NAME" \
    --epochs 1 \
    --batch_size 6 \
    --learning_rate 1e-4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --save_steps 3 \
    --concept_token "<mensafood>"

echo "[OK] Training complete!"

# Inference
echo "[INFER] Starting inference tests..."

echo "[INFER] Generating Currywurst image..."
python3 infer_enhanced.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "Currywurst oder planted Currywurst Pommes frites" \
    --steps 50 || echo "[!] Currywurst inference failed, continuing..."

echo "[INFER] Generating Cordon Bleu image..."
python3 infer_enhanced.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "Cordon Bleu Vom Schwein Zitronenjus Kartoffelbrei Karotten Erbsengemuse" \
    --steps 50 || echo "[!] Cordon Bleu inference failed, continuing..."

echo "[OK] Inference complete!"

# FID test
echo "[EVAL] Running FID evaluation..."
chmod +x ./calculate_fid.sh

./calculate_fid.sh "$EXPERIMENT_NAME" || echo "[!] FID calculation failed, continuing..."

echo "============================================================"
echo "[RESULT] Generated images in ./experiments/$EXPERIMENT_NAME/outputs/"
echo "[RESULT] LoRA weights in ./experiments/$EXPERIMENT_NAME/lora_weights/"
echo "[RESULT] FID results in ./experiments/$EXPERIMENT_NAME/fid_results.txt"
echo "============================================================"
