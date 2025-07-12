# Mensa LoRA Training - Experiment Structure

## Overview
This project fine-tunes Stable Diffusion v1.4 with LoRA for generating German Mensa food images. The training script uses a clean experiment-based organization system.

## Experiment Structure
When you run training with `--experiment_name "my_experiment"`, the following directory structure is created:

```
experiments/
├── my_experiment/
│   ├── lora_weights/          # LoRA model weights and checkpoints
│   │   ├── pytorch_lora_weights_best.bin
│   │   ├── pytorch_lora_weights_final.bin
│   │   ├── pytorch_lora_weights_epoch_4.bin
│   │   └── training_config.json
│   └── outputs/               # Generated inference samples
│       ├── inference_sample_1.png
│       └── inference_sample_2.png
```

## Quick Start

### 1. Training
```bash
# Local training
python3 train_lora_enhanced.py \
    --experiment_name "my_food_experiment" \
    --dataset_csv ./dataset/dataset.csv \
    --epochs 8 \
    --batch_size 1 \
    --learning_rate 5e-5

# SLURM cluster training
sbatch slurm_train.sh
```

### 2. Inference
```bash
python3 infer_enhanced.py \
    --prompt "currywurst german sausage with french fries on plate in cafeteria tray" \
    --lora_weights ./experiments/my_food_experiment/lora_weights \
    --output ./my_currywurst.png
```

## Key Features

- **English Prompts**: Uses English prompts for better CLIP model understanding
- **Memory Optimized**: Configured for RTX 2080 (11GB VRAM) with batch_size=1
- **Auto-organization**: Each experiment gets its own clean folder structure
- **Checkpoint Saving**: Saves best model, final model, and periodic checkpoints
- **Training Config**: Saves complete training configuration for reproducibility

## Default Training Settings

- **Base Model**: CompVis/stable-diffusion-v1-4
- **LoRA Rank**: 4 (low memory usage)
- **LoRA Alpha**: 16
- **Learning Rate**: 5e-5
- **Batch Size**: 1 (RTX 2080 optimized)
- **Epochs**: 8
- **Save Steps**: 4 (checkpoint every 4 epochs)

## Dataset Format

The training expects a CSV file with columns:
- `image_path`: Path to food image
- `caption`: English description of the dish

Current dataset focuses on the top 2 Mensa dishes:
1. "currywurst german sausage with french fries on plate in cafeteria tray"
2. "pasta dish with sauce and toppings on plate in cafeteria tray"

## Troubleshooting

### Memory Issues
- Reduce `--batch_size` to 1
- Lower `--resolution` from 512 to 256
- Reduce `--lora_r` from 4 to 2

### Training Not Converging
- Increase `--epochs` 
- Adjust `--learning_rate` (try 1e-4 or 1e-5)
- Check your dataset quality and captions

### SLURM Issues
- Check logs in `logs/train_*.out` and `logs/train_*.err`
- Verify GPU availability: `squeue -u $USER`
- Check CUDA setup in SLURM environment
