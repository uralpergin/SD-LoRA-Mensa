#!/bin/bash
#SBATCH --job-name=vanilla-sd
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=30:00

echo "============================================================"
echo "               MENSA VANILLA SD - INFERENCE                 "
echo "============================================================"
echo "[JOB]  Job ID : $SLURM_JOB_ID"
echo "[NODE] Node   : $(hostname)"

# ---------------- Configuration ----------------
EXPERIMENT_NAME="${1:-vanilla_default}"          # 1st arg: experiment name
ARGC=$#
LAST_ARG="${!#}"

# Default num_images
NUM_IMAGES=3

# Determine if the last arg is an integer -> then it's num_images
if [[ $ARGC -ge 2 && "$LAST_ARG" =~ ^[0-9]+$ ]]; then
  NUM_IMAGES="$LAST_ARG"
  # Prompt words are args 2..(last-1)
  if (( ARGC > 2 )); then
    PROMPT_WORDS=("${@:2:ARGC-2}")
  else
    PROMPT_WORDS=()
  fi
else
  # Prompt words are args 2..end
  if (( ARGC >= 2 )); then
    PROMPT_WORDS=("${@:2}")
  else
    PROMPT_WORDS=()
  fi
fi

# Join prompt words (if any) into a single string
PROMPT_ARG="${PROMPT_WORDS[*]}"

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
export CUDA_LAUNCH_BLOCKING=1    
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
import torch
print(f"[DEPS] PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
PY

# ---------------- Repository root ---------------
cd /work/dlclarge2/matusd-test/SD-LoRA-Mensa

# ---------------- Inference ---------------------
INFER_SCRIPT="src/infer_vanilla_sd.py"   

# If a prompt override is provided, use it; otherwise use defaults.
if [[ -n "$PROMPT_ARG" ]]; then
  echo "[CFG] Using single prompt from args."
  PROMPTS=("$PROMPT_ARG")
else
  echo "[CFG] Using default prompt list."
  PROMPTS=(
    "Minced steak Bernese style with pepper, mashed potatoes, carrots and peas"
    "Asparagus in white sauce sauteed potatoes"
    "Ravioli with herb pesto on tomato lentil stew"
    "Spätzle with beef goulash and sauteed green beans"
  )
fi

STEPS=60
GUIDANCES=(3.5 7.5)

echo "[INFER] Steps       : $STEPS"
echo "[INFER] Num images  : $NUM_IMAGES"
echo "[INFER] Guidances   : ${GUIDANCES[*]}"

for PROMPT in "${PROMPTS[@]}"; do
  echo "------------------------------------------------------------"
  echo "[INFER] Prompt: $PROMPT"
  for GUIDANCE in "${GUIDANCES[@]}"; do
    echo "[INFER]  ➤ guidance=$GUIDANCE"
    python3 "$INFER_SCRIPT" \
      --experiment_name "$EXPERIMENT_NAME" \
      --prompt "$PROMPT" \
      --steps "$STEPS" \
      --guidance "$GUIDANCE" \
      --num_images "$NUM_IMAGES" || echo "[!] Inference failed for '$PROMPT' (guidance=$GUIDANCE)"
  done
done

echo "============================================================"
echo "[OK] Inference complete!"
echo "[RESULT] Images saved under ./experiments/$EXPERIMENT_NAME/outputs/"
echo "============================================================"
