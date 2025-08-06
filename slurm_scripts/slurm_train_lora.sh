#!/bin/bash
#SBATCH --job-name=lora
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=5:30:00

echo "============================================================"
echo "              MENSA LORA TRAINING JOB"
echo "============================================================"
echo "[JOB] Job ID: $SLURM_JOB_ID"
echo "[NODE] Node : $(hostname)"

# ---------------- Configuration ----------------
EXPERIMENT_NAME="${1:-experiment_default}"
ROOT_DIR="/work/dlclarge2/matusd-lora/mensa-lora" # TODO: change to correct path
LOG_DIR="${ROOT_DIR}/logs/${EXPERIMENT_NAME}"
FID_DIR="${ROOT_DIR}/fid_original_images"

mkdir -p "${LOG_DIR}" "${FID_DIR}" logs

exec 1> >(tee "${LOG_DIR}/train_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/train_${SLURM_JOB_ID}.err" >&2)

echo "[EXP] Experiment : $EXPERIMENT_NAME"
echo "[LOG] Directory  : ${LOG_DIR}/"
echo "[OUT] Stdout log : ${LOG_DIR}/train_${SLURM_JOB_ID}.out"
echo "[ERR] Stderr log : ${LOG_DIR}/train_${SLURM_JOB_ID}.err"

# ---------------- Inference flag (last argument) ------------
ARGC=$#
LAST_ARG="${!#}"
RUN_INFER=0  # default: skip inference
if [[ $ARGC -ge 2 && "$LAST_ARG" == "--inference" ]]; then
  RUN_INFER=1
fi
echo "[CFG] Inference after training: $([[ $RUN_INFER -eq 1 ]] && echo ENABLED || echo DISABLED)"

# ---------------- GPU Info ----------------
echo "[GPU] GPU Information:"
nvidia-smi || echo "[WARNING] nvidia-smi command failed"

echo "[GPU] Checking GPU stability..."
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits || echo "[WARNING] GPU query failed"

# ---------------- CUDA Environment ----------------
echo "[CUDA] Setting up CUDA environment..."
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"

# ---------------- Virtual Environment ----------------
VENV=~/venvs/mensa
echo "[VENV] Activating virtual environment..."
[ ! -d "$VENV" ] && python3 -m venv "$VENV"
source "$VENV/bin/activate"

# ---------------- Dependencies ----------------
echo "[DEPS] Installing dependencies..."
pip install --upgrade --quiet pip
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --quiet transformers diffusers accelerate peft datasets
pip install --quiet 'numpy<2' scipy pandas scikit-learn matplotlib seaborn
pip install --quiet optuna plotly joblib kaleido Pillow tqdm

echo "[DEPS] Setup complete!"
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"

# ---------------- Training ----------------
cd "$ROOT_DIR" || { echo "[!] Could not cd to $ROOT_DIR"; exit 1; }

if command -v nvidia-smi &>/dev/null; then
    echo "[MEMORY] Available GPU memory before training:"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits || echo "[WARNING] Could not check GPU memory"
fi

python3 src/train_lora.py \
    --dataset_csv ./dataset/dataset.csv \
    --experiment_name "$EXPERIMENT_NAME" \
    --epochs 2 \
    --batch_size 6 \
    --learning_rate 6.218704727769077e-05 \
    --lora_r 4 \
    --lora_alpha 64 \
    --save_steps 5 \
    --concept_token "<mensafood>" \
    --duplicate 1 \
    --weight_decay 0.001 \
    --warmup_ratio 0.2 \
    --lora_dropout 0.1

echo "[OK] Training complete!"

# ---------------- Optional Inference ----------------
if [[ $RUN_INFER -eq 1 ]]; then
  echo "[INFER] Starting inference tests..."

  declare -a PROMPTS=(
      "Minced steak Bernese style with pepper, mashed potatoes, carrots and peas"
      "Asparagus in white sauce sauteed potatoes"
      "Ravioli with herb pesto on tomato lentil stew"
      "Spätzle with beef goulash and sauteed green beans"
  )

  for PROMPT in "${PROMPTS[@]}"; do
      echo "[INFER] Generating: $PROMPT"
      for GUIDANCE in 3.5 7.5; do
          python3 src/infer_lora.py \
              --experiment_name "$EXPERIMENT_NAME" \
              --prompt "$PROMPT" \
              --steps 60 \
              --guidance "$GUIDANCE" \
              --num_images 5 || echo "[!] Inference failed for: $PROMPT (guidance=$GUIDANCE)"
      done
  done

  echo "[OK] Inference complete!"
  echo "============================================================"
  echo "[RESULT] LoRA weights in ./experiments/$EXPERIMENT_NAME/lora_weights/"
  echo "[RESULT] Generated images in ./experiments/$EXPERIMENT_NAME/outputs/"
  echo "============================================================"
else
  echo "[SKIP] Inference skipped (use '--inference' to enable)."
  echo "============================================================"
  echo "[RESULT] LoRA weights in ./experiments/$EXPERIMENT_NAME/lora_weights/"
  echo "============================================================"
fi
