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
echo "Logs directory: ${LOG_DIR}/"
echo "Output log: ${LOG_DIR}/train_${SLURM_JOB_ID}.out"
echo "Error log: ${LOG_DIR}/train_${SLURM_JOB_ID}.err"

# Create directories
mkdir -p logs

echo "[GPU] GPU Information:"
nvidia-smi

echo "[GPU] Checking GPU stability..."
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

echo "[CUDA] Setting up CUDA environment..."
export CUDA_LAUNCH_BLOCKING=1
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"

echo "[DEPS] Installing remaining dependencies for LoRA training..."
pip3 install --user peft accelerate datasets

echo "[TRAIN] Starting LoRA training..."
cd /work/dlclarge2/ceylanb-DL_Lab_Project/mensa-lora

# Run training
python3 train_lora_enhanced.py \
    --dataset_csv ./dataset/dataset.csv \
    --experiment_name "$EXPERIMENT_NAME" \
    --epochs 15 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --lora_r 32 \
    --lora_alpha 128 \
    --save_steps 3 \
    --cfg_weight 0.3

echo "[OK] Training complete!"

# Test inference with both dishes (with error handling)
echo "[INFER] Starting inference tests..."

echo "[INFER] Generating Currywurst image..."
python3 infer_enhanced.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "Currywurst oder planted Currywurst Pommes frites" \
    --steps 50 || echo "[!] Currywurst inference failed, continuing..."

echo "[INFER] Generating Pasta image..."
python3 infer_enhanced.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "Pasta-Kreationen aus unserer eigenen Pasta-Manufaktur mit verschiedenen Saucen und Toppings" \
    --steps 50 || echo "[!] Pasta inference failed, continuing..."

echo "[OK] Inference complete!"

# Run FID evaluation directly with the conda environment
echo "[EVAL] Running FID evaluation..."
chmod +x ./calculate_fid.sh
# Pass the current environment to the evaluation script
./calculate_fid.sh "$EXPERIMENT_NAME" || echo "[!] FID calculation failed, continuing..."

echo "============================================================"
echo "[RESULT] Generated images in ./experiments/$EXPERIMENT_NAME/outputs/"
echo "[RESULT] LoRA weights in ./experiments/$EXPERIMENT_NAME/lora_weights/"
echo "[RESULT] FID results in ./experiments/$EXPERIMENT_NAME/fid_results.txt"
echo "============================================================"
