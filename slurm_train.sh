#!/bin/bash
#SBATCH --job-name=mensa-lora-train
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=01:30:00

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
echo "Logs directory: ${LOG_DIR}/"
echo "Output log: ${LOG_DIR}/train_${SLURM_JOB_ID}.out"
echo "Error log: ${LOG_DIR}/train_${SLURM_JOB_ID}.err"

# Create directories
mkdir -p logs

echo "[GPU] GPU Information:"
nvidia-smi

echo "[CUDA] Setting up CUDA environment..."
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"

echo "[TRAIN] Starting LoRA training..."
cd /work/dlclarge2/ceylanb-DL_Lab_Project/mensa-lora

# Run training
python3 train_lora_enhanced.py \
    --dataset_csv ./dataset/dataset.csv \
    --experiment_name "$EXPERIMENT_NAME" \
    --epochs 8 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --lora_r 4 \
    --lora_alpha 16 \
    --save_steps 4

echo "[✓] Training complete!"

# Test inference with both dishes (with error handling)
echo "[INFER] Starting inference tests..."

python3 infer_enhanced.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "currywurst german sausage with french fries on plate in cafeteria tray" \
    --steps 25 \
    --seed 42 || echo "[!] Inference 1 failed, continuing..."

python3 infer_enhanced.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "pasta dish with sauce and toppings on plate in cafeteria tray" \
    --steps 25 \
    --seed 42 || echo "[!] Inference 2 failed, continuing..."

echo "[✓] Inference complete!"

echo "============================================================"
echo "[RESULT] Generated images in ./experiments/$EXPERIMENT_NAME/outputs/"
echo "[RESULT] LoRA weights in ./experiments/$EXPERIMENT_NAME/lora_weights/"
echo "============================================================"
