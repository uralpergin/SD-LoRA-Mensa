#!/bin/bash
#SBATCH --job-name=lora-infer
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=15:00

echo "============================================================"
echo "           MENSA LORA — INFERENCE"
echo "============================================================"
echo "[JOB] Job ID : $SLURM_JOB_ID"
echo "[NODE] Node  : $(hostname)"

# ---------- Configuration ----------
EXPERIMENT_NAME="${1:-experiment_default}"   # 1st arg: experiment name
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

# Join prompt words; fall back to default if empty
PROMPT_ARG="${PROMPT_WORDS[*]}"
if [[ -n "$PROMPT_ARG" ]]; then
  PROMPT="$PROMPT_ARG"
else
  PROMPT="Grilled eggplant with spatzle"
fi

ROOT_DIR="/work/dlclarge2/matusd-lora/mensa-lora" # TODO: change to correct path
LOG_DIR="${ROOT_DIR}/logs2/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

exec 1> >(tee "${LOG_DIR}/infer_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/infer_${SLURM_JOB_ID}.err" >&2)

echo "[EXP] Experiment : $EXPERIMENT_NAME"
echo "[LOG] Directory  : ${LOG_DIR}/"

# ---------- GPU + CUDA ----------
echo "[GPU] GPU info:"
nvidia-smi || echo "[WARNING] nvidia-smi not found"

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"

# ---------- Python env ----------
VENV=~/venvs/mensa
echo "[VENV] Activating venv..."
source "$VENV/bin/activate"

# ---------- Inference ----------
cd "$ROOT_DIR" || { echo "[!] Could not cd to $ROOT_DIR"; exit 1; }

GUIDANCES=(3.5 7.5)
STEPS=60

echo "[INFER] Prompt    : $PROMPT"
echo "[INFER] Steps     : $STEPS"
echo "[INFER] NumImages : $NUM_IMAGES"
echo "[INFER] Guidances : ${GUIDANCES[*]}"

for GUIDANCE in "${GUIDANCES[@]}"; do
  echo "------------------------------------------------------------"
  echo "[INFER] Running with guidance: $GUIDANCE"
  python3 src/infer_lora.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --prompt "$PROMPT" \
    --steps "$STEPS" \
    --guidance "$GUIDANCE" \
    --num_images "$NUM_IMAGES" || { echo "[!] Inference failed for guidance $GUIDANCE"; exit 1; }
done

echo "============================================================"
echo "[RESULT] Images: ./experiments/$EXPERIMENT_NAME/outputs/"
echo "============================================================"