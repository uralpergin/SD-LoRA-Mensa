#!/bin/bash
#SBATCH --job-name=mensa-lora-train
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=05:30:00

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
nvidia-smi || echo "[WARNING] nvidia-smi command failed, but continuing"

echo "[GPU] Checking GPU stability..."
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits || echo "[WARNING] GPU query failed, but continuing"

echo "[CUDA] Setting up CUDA environment..."
# Clear any existing GPU cache
if command -v nvidia-smi &>/dev/null; then
    echo "[CUDA] Clearing GPU cache"
    cuda-empty-cache &>/dev/null || echo "[WARNING] Could not clear CUDA cache"
fi

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1 # To see CUDA errors, delete for full power
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"

echo "[DEPS] Installing remaining dependencies for LoRA training..."
# Silence pip
export PIP_DISABLE_PIP_VERSION_CHECK=1
pip3 install --user --disable-pip-version-check --quiet peft accelerate datasets \
  > /dev/null 2>&1

cd /work/dlclarge2/ceylanb-DL_Lab_Project/mensa-lora

# Check GPU memory before training
if command -v nvidia-smi &>/dev/null; then
    echo "[MEMORY] Available GPU memory before training:"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits || echo "[WARNING] Could not check GPU memory"
fi

# Run training with reduced batch size to prevent OOM
python3 lora_train.py \
    --dataset_csv ./dataset/dataset.csv \
    --experiment_name "$EXPERIMENT_NAME" \
    --epochs 50 \
    --batch_size 6 \
    --learning_rate 5e-5 \
    --lora_r 8 \
    --lora_alpha 32 \
    --save_steps 10 \
    --concept_token "<mensafood>"

echo "[OK] Training complete!"

# Inference
echo "[INFER] Starting inference tests..."

echo "[INFER] Generating Steak image..."
python3 inferance.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "Minced steak Bernese style with pepper, mashed potatoes, carrots and peas" \
    --steps 60 \
    --guidance 3.5 || echo "[!] Steak inference failed, continuing..."

python3 inferance.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "Minced steak Bernese style with pepper, mashed potatoes, carrots and peas" \
    --steps 60 \
    --guidance 5 || echo "[!] Steak inference failed, continuing..."

echo "[INFER] Generating Potato Pancakes image..."

python3 inferance.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "Potato pancakes with peas, guacamole and herbs yogurt dip" \
    --steps 60 \
    --guidance 3.5 || echo "[!] Potato Pancakes inference failed, continuing..."

python3 inferance.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "Potato pancakes with peas, guacamole and herbs yogurt dip" \
    --steps 60 \
    --guidance 5 || echo "[!] Potato Pancakes inference failed, continuing..."

echo "[OK] Inference complete!"

echo "============================================================"
echo "[RESULT] Generated images in ./experiments/$EXPERIMENT_NAME/outputs/"
echo "[RESULT] LoRA weights in ./experiments/$EXPERIMENT_NAME/lora_weights/"
echo "============================================================"
