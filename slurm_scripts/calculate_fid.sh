#!/bin/bash
#SBATCH --job-name=mensa-lora-train
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=02:30:00

set -euo pipefail
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

cd /work/dlclarge2/erginu-dl_lab_project/mensa-lora # Change to your project directory

pip install --quiet torch-fidelity # make sure this is installed
# TODO: change the path to image folders
# Manually put the original images in the following folders:
# asparagus, ravioli, spatzle, steak
python3 src/calculate_fid.py \
  --real_dirs \
    /work/dlclarge2/erginu-dl_lab_project/mensa-lora/fid_original_images/asparagus \
    /work/dlclarge2/erginu-dl_lab_project/mensa-lora/fid_original_images/ravioli \
    /work/dlclarge2/erginu-dl_lab_project/mensa-lora/fid_original_images/spatzle \
    /work/dlclarge2/erginu-dl_lab_project/mensa-lora/fid_original_images/steak \
  --gen_dirs \
    /work/dlclarge2/erginu-dl_lab_project/mensa-lora/generated_images/asparagus \
    /work/dlclarge2/erginu-dl_lab_project/mensa-lora/generated_images/ravioli \
    /work/dlclarge2/erginu-dl_lab_project/mensa-lora/generated_images/spatzle \
    /work/dlclarge2/erginu-dl_lab_project/mensa-lora/generated_images/steak \
  --output "fid_results.txt"
