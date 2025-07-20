#!/bin/bash
#SBATCH --job-name=mensa-lora-train
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=02:30:00

# Get the experiment name
EXPERIMENT_NAME=${1:-"experiment_default"}
echo "Calculating FID for experiment: $EXPERIMENT_NAME"

# CUDA env setup
export CUDA_LAUNCH_BLOCKING=1
source /etc/cuda_env
cuda12.6
echo "[CUDA] CUDA_HOME: $CUDA_HOME"

export PATH="$HOME/.local/bin:$PATH"

# Define dirs
REAL_IMAGES_DIR="./dataset/images"
GEN_IMAGES_DIR="./experiments/${EXPERIMENT_NAME}/outputs"
RESULT_FILE="./experiments/${EXPERIMENT_NAME}/fid_results.txt"

TEMP_REAL_DIR="./temp_real_images_$$"
TEMP_GEN_DIR="./temp_gen_images_$$"
mkdir -p "$TEMP_REAL_DIR" "$TEMP_GEN_DIR"

# Check if images for comparison exist
if [ ! -d "$REAL_IMAGES_DIR" ]; then
    echo "ERROR: Real images directory missing: $REAL_IMAGES_DIR"
    exit 1
fi

if [ ! -d "$GEN_IMAGES_DIR" ]; then
    echo "ERROR: Generated images directory missing: $GEN_IMAGES_DIR"
    exit 1
fi

# Count images before copying
real_count=$(find "$REAL_IMAGES_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)
gen_count=$(find "$GEN_IMAGES_DIR" -type f -iname "*.png" | wc -l)

echo "Found $real_count real images in $REAL_IMAGES_DIR"
echo "Found $gen_count generated images in $GEN_IMAGES_DIR"

if [ "$real_count" -eq 0 ]; then
    echo "ERROR: No real images found"
    exit 1
fi

if [ "$gen_count" -eq 0 ]; then
    echo "ERROR: No generated images found. Make sure inference completed successfully."
    exit 1
fi

# Copy image files
echo "Copying real images..."
find "$REAL_IMAGES_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) \
    -exec cp {} "$TEMP_REAL_DIR" \;

echo "Copying generated images..."
find "$GEN_IMAGES_DIR" -type f -iname "*.png" \
    -exec cp {} "$TEMP_GEN_DIR" \;

# Verify copies
copied_real=$(ls "$TEMP_REAL_DIR" | wc -l)
copied_gen=$(ls "$TEMP_GEN_DIR" | wc -l)

echo "Copied $copied_real real images to temp directory"
echo "Copied $copied_gen generated images to temp directory"

if [ "$copied_real" -eq 0 ] || [ "$copied_gen" -eq 0 ]; then
    echo "ERROR: Failed to copy images to temporary directories"
    rm -rf "$TEMP_REAL_DIR" "$TEMP_GEN_DIR"
    exit 1
fi

# Prepare output dir
mkdir -p "$(dirname "$RESULT_FILE")"
{
  echo "FID RESULTS FOR EXPERIMENT: $EXPERIMENT_NAME"
  echo "Date: $(date)"
  echo "----------------------------------------"
} > "$RESULT_FILE"

# Install torch-fidelity
if ! command -v fidelity &>/dev/null; then
  echo "Installing torch-fidelity into \$HOME/.local ..."
  pip3 install --user --quiet torch-fidelity
fi

# Run FID
echo "Calculating FID score..."
fidelity --gpu 0 --fid \
   --batch-size 1 \
   --num-workers 0 \
   --input1 "$TEMP_REAL_DIR" \
   --input2 "$TEMP_GEN_DIR" \
   | tee -a "$RESULT_FILE"

# Cleanup
echo "Cleaning up temporary folders..."
rm -rf "$TEMP_REAL_DIR" "$TEMP_GEN_DIR"

echo "FID calculation complete. Results: $RESULT_FILE"
