#!/bin/bash
#SBATCH --job-name=onlySD
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=00:30:00

# Script to generate images with only the base Stable Diffusion model
echo "==============================================================="
echo "Generating images with vanilla Stable Diffusion (no LoRA)"
echo "==============================================================="

# Set up environment
echo "[SETUP] Setting up environment"
export CUDA_LAUNCH_BLOCKING=1
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"

# Create directory structure
mkdir -p ./experiments/onlyStableDiffusion/outputs

# Run multiple test prompts to generate a good set of comparison images
echo "[INFER] Generating Currywurst image with vanilla SD..."
python3 infer_noLoRA.py \
    --prompt "Minced steak Bernese style with pepper, mashed potatoes, carrots and peas" \
    --steps 50 \

echo "[INFER] Generating Pasta image with vanilla SD..."
python3 infer_noLoRA.py \
    --prompt "Vegetable gnocchi with pink sauce and cheese topping" \
    --steps 50 \

echo "[OK] All vanilla Stable Diffusion images generated"
echo "Results saved in ./experiments/onlyStableDiffusion/outputs/"
echo "==============================================================="
