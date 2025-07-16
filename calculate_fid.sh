#!/bin/bash
#SBATCH --job-name=mensa-lora-train
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=02:30:00
# Simple FID calculation between real dataset images and generated images

# Get the experiment name
EXPERIMENT_NAME=${1:-"experiment_default"}
echo "Calculating FID for experiment: $EXPERIMENT_NAME"

# Setup CUDA environment
export CUDA_LAUNCH_BLOCKING=1
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"

# Define paths
REAL_IMAGES_DIR="./dataset/images"
GEN_IMAGES_DIR="./experiments/${EXPERIMENT_NAME}/outputs"
RESULT_FILE="./experiments/${EXPERIMENT_NAME}/fid_results.txt"

# Create temporary directories for filtered images
TEMP_REAL_DIR="./temp_real_images_$$"  # Using PID in temp dir name to avoid conflicts
TEMP_GEN_DIR="./temp_gen_images_$$"

mkdir -p "$TEMP_REAL_DIR" "$TEMP_GEN_DIR"

# Check if directories exist
if [ ! -d "$REAL_IMAGES_DIR" ]; then
    echo "ERROR: Real images directory not found: $REAL_IMAGES_DIR"
    exit 1
fi

if [ ! -d "$GEN_IMAGES_DIR" ]; then
    echo "ERROR: Generated images directory not found: $GEN_IMAGES_DIR"
    exit 1
fi

# Filter out non-image files
echo "Filtering image files for FID calculation..."
find "$REAL_IMAGES_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) -exec cp {} "$TEMP_REAL_DIR" \;
find "$GEN_IMAGES_DIR" -type f -name "*.png" -exec cp {} "$TEMP_GEN_DIR" \;

echo "============================================================"
echo "            FID CALCULATION FOR MENSA FOOD IMAGES"
echo "============================================================"

# Create output directory if needed
mkdir -p "$(dirname "$RESULT_FILE")"

# Save basic info
echo "FID RESULTS FOR EXPERIMENT: $EXPERIMENT_NAME" > $RESULT_FILE
echo "Date: $(date)" >> $RESULT_FILE
echo "----------------------------------------" >> $RESULT_FILE

# Ensure torch-fidelity is installed
pip3 install --quiet torch-fidelity

# Run FID calculation
echo "Calculating FID score..."
python3 -m torch_fidelity --gpu 0 --fid --input1 "$TEMP_REAL_DIR" --input2 "$TEMP_GEN_DIR" | tee -a $RESULT_FILE

# Clean up temporary directories
echo "Cleaning up..."
rm -rf "$TEMP_REAL_DIR" "$TEMP_GEN_DIR"

echo "============================================================"
echo "FID calculation complete - Results saved to $RESULT_FILE"
echo "============================================================"
