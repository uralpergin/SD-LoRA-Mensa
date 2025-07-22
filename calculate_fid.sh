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

# Create temporary directories with unique IDs to prevent conflicts
TEMP_REAL_DIR="./temp_real_images_$$"
TEMP_GEN_DIR="./temp_gen_images_$$"
mkdir -p "$TEMP_REAL_DIR" "$TEMP_GEN_DIR"

echo "Creating temporary directories:"
echo "- Real images: $TEMP_REAL_DIR"
echo "- Generated images: $TEMP_GEN_DIR"

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

# Match images from dataset.csv to ensure fair comparison
DATASET_CSV="./dataset/dataset.csv"
if [ -f "$DATASET_CSV" ]; then
    echo "Using dataset.csv to match real and generated images..."
    
    # Extract basenames from the CSV file (assuming image_path column exists)
    echo "Extracting image basenames from CSV..."
    IMAGE_BASENAMES=$(cut -d',' -f1 "$DATASET_CSV" | grep -v "image_path" | xargs -I{} basename {} | sed 's/\.[^.]*$//')
    
    # Copy only the images used in training
    echo "Copying matched real images..."
    for basename in $IMAGE_BASENAMES; do
        find "$REAL_IMAGES_DIR" -type f \( -iname "${basename}.*" \) -exec cp {} "$TEMP_REAL_DIR" \; 2>/dev/null
    done
    
    echo "Copying generated images..."
    # For generated images, we take all since they should correspond to our training set
    find "$GEN_IMAGES_DIR" -type f -iname "*.png" -exec cp {} "$TEMP_GEN_DIR" \;
else
    # Fallback to copying all images if CSV doesn't exist
    echo "Warning: dataset.csv not found. Falling back to comparing all images."
    echo "Copying all real images..."
    find "$REAL_IMAGES_DIR" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) \
        -exec cp {} "$TEMP_REAL_DIR" \;
        
    echo "Copying all generated images..."
    find "$GEN_IMAGES_DIR" -type f -iname "*.png" \
        -exec cp {} "$TEMP_GEN_DIR" \;
fi

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

# Check if we have enough images for a meaningful comparison
if [ "$copied_real" -lt 10 ] || [ "$copied_gen" -lt 10 ]; then
    echo "WARNING: Very few images for FID calculation. Results may not be statistically significant."
    echo "WARNING: FID calculation works best with at least 10 images in each set." | tee -a "$RESULT_FILE"
fi

# Run FID
echo "Calculating FID score between $copied_real real images and $copied_gen generated images..."
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
