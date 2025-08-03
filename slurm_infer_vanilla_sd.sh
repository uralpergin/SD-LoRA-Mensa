#!/bin/bash
#SBATCH --job-name=vanilla-sd
#SBATCH --partition=testdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=30:00

echo "============================================================"
echo "               MENSA VANILLA SD INFERENCE JOB"
echo "============================================================"
echo "[JOB]  Job ID : $SLURM_JOB_ID"
echo "[NODE] Node   : $(hostname)"

# ---------------- Configuration ----------------
EXPERIMENT_NAME="${1:-vanilla_default}"          # e.g. vanilla_20250803
LOG_DIR="logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

# Redirect stdout / stderr to experiment-specific log files
exec 1> >(tee  "${LOG_DIR}/infer_${SLURM_JOB_ID}.out")
exec 2> >(tee  "${LOG_DIR}/infer_${SLURM_JOB_ID}.err" >&2)

echo "[EXP] Experiment      : $EXPERIMENT_NAME"
echo "[LOG] Logs directory  : ${LOG_DIR}/"
echo "[OUT] Stdout log      : ${LOG_DIR}/infer_${SLURM_JOB_ID}.out"
echo "[ERR] Stderr log      : ${LOG_DIR}/infer_${SLURM_JOB_ID}.err"
echo "------------------------------------------------------------"

# ---------------- Hardware check ----------------
echo "[GPU] GPU Information:"
nvidia-smi || echo "[WARNING] nvidia-smi unavailable, continuing"

# ---------------- CUDA setup --------------------
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1     # Comment out for max speed once stable
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"

# ---------------- Python environment ------------ 
VENV=~/venvs/mensa
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Dependencies (runs fast if already installed)
pip install --upgrade --quiet pip
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --quiet diffusers transformers accelerate Pillow tqdm "numpy<2"

python - <<'PY'
import torch, diffusers, PIL, sys, platform, subprocess, json, datetime
print(f"[DEPS] PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}")
PY

# ---------------- Repository root ---------------
cd /work/dlclarge2/matusd-lora/mensa-lora

# ---------------- Inference ---------------------
INFER_SCRIPT="infer_vanilla_sd.py"   

PROMPTS=(
  "Minced steak Bernese style with pepper, mashed potatoes, carrots and peas"
  "Asparagus in white sauce sauteed potatoes"
  "Ravioli with herb pesto on tomato lentil stew"
  "Spätzle with beef goulash and sauteed green beans"
)

for PROMPT in "${PROMPTS[@]}"; do
  echo "------------------------------------------------------------"
  echo "[INFER] Prompt: $PROMPT"
  for GUIDANCE in 3.5 7.5; do
    echo "[INFER]  ➤ guidance=$GUIDANCE"
    python3 "$INFER_SCRIPT" \
      --experiment_name "$EXPERIMENT_NAME" \
      --prompt "$PROMPT" \
      --steps 60 \
      --guidance "$GUIDANCE" \
      --num_images 5 || echo "[!] Inference failed for '$PROMPT' (guidance=$GUIDANCE)"
  done
done

echo "============================================================"
echo "[OK] Inference complete!"
echo "[RESULT] Images saved under ./experiments/$EXPERIMENT_NAME/outputs/"
echo "============================================================"
