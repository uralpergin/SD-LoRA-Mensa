"""
Stable Diffusion with LoRA fine-tune Training for Mensa Food Generation
"""

import os
import pandas as pd
import torch
from PIL import Image
from datasets import Dataset
from peft import LoraConfig, get_peft_model_state_dict
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.training_utils import compute_snr, cast_training_params
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import gc

def setup_memory_optimization():
    """Setup memory optimization and get current state"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear cache
        torch.backends.cuda.matmul.allow_tf32 = True # Efficent matrix multiplication
        torch.backends.cudnn.allow_tf32 = True # Faster convolutions
        print(f"[INFO] CUDA Device: {torch.cuda.get_device_name()}")
        print(f"[INFO] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def load_from_csv(csv_path):
    """
    Load images and prompts from CSV

    csv_path: Path to the CSV file containing image paths and prompts
    returns: List of dictionaries with image and text data
    """
    df = pd.read_csv(csv_path)
    samples = []
    missing_count = 0

    print(f"[LOAD] Loading dataset from {csv_path}")
    print(f"[LOAD] Total entries: {len(df)}")

    for idx, row in df.iterrows():
        image_path = row["image_path"]
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"[x] Image not found: {image_path}")
            missing_count += 1
            continue

        # Test to see if english text works better.
        description = row["description"]
        
        if "Currywurst" in description: # for testing
            prompt = "currywurst german sausage with french fries on plate in cafeteria tray"
        elif "Pasta" in description: # for testing
            prompt = "pasta dish with sauce and toppings on plate in cafeteria tray"
        else:
            prompt = row["description"]  # original german description
        #print(prompt)
        
        try:
            image = Image.open(image_path).convert("RGB")
            samples.append({"image": image, "label": prompt})
        except Exception as e:
            print(f"[x] Failed to load image {image_path}: {e}")
            missing_count += 1

    print(f"[✓] Loaded {len(samples)} images successfully")
    if missing_count > 0:
        print(f"[!] {missing_count} images were missing or failed to load")
    
    return samples

def tokenize_and_transform(sample, tokenizer, resolution):
    """
    Tokenize text and transform images
    sample: Dictionary with 'image' and 'text' keys
    tokenizer: CLIPTokenizer instance
    resolution: Image resolution for training
    """
    # Tokenize text
    encoding = tokenizer(
        sample["label"], # meal label 
        padding="max_length", # pad each text to max length
        truncation=True, # if the text is to long cut off
        max_length=tokenizer.model_max_length, 
        return_tensors="pt", # return PyTorch tensors
    )
    
    sample["input_ids"] = encoding["input_ids"].squeeze(0) # Tokenized text
    
    # Transform image
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=Image.BILINEAR),
        transforms.CenterCrop(resolution), # resolution is 512x512 since stable diffusion v1.4 was trained on 512x512
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    sample["pixel_values"] = transform(sample["image"]) # transformed image
    return sample

def save_training_config(output_dir, config):
    """Save training configuration for reproducibility"""
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[SAVE] Training config saved to: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='LoRA Training for Mensa Food Generation')
    parser.add_argument('--dataset_csv', default="./dataset/dataset.csv", 
                       help='Path to dataset CSV file')
    parser.add_argument('--experiment_name', default="experiment_default", 
                       help='Name for this experiment')
    parser.add_argument('--pretrained_model', default="CompVis/stable-diffusion-v1-4", 
                       help='Pretrained Stable Diffusion model')
    parser.add_argument('--resolution', type=int, default=512, 
                       help='Training resolution')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Training batch size') # Keep at 1 for 12GB GPU
    parser.add_argument('--epochs', type=int, default=8, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, 
                       help='Learning rate')
    # LoRA Rank:
    # Larger values allow more adaptation (capacity of the adaptation increase) but require more VRAM
    parser.add_argument('--lora_r', type=int, default=4, 
                       help='LoRA rank') 
    # LoRA Alpha:
    # Larger values increase adaptation strength (output affects from tuning more) but require more VRAM
    parser.add_argument('--lora_alpha', type=int, default=16, 
                       help='LoRA alpha')
    parser.add_argument('--save_steps', type=int, default=4, 
                       help='Save model every N epochs')
    
    args = parser.parse_args()
    
    # Setup memory optimization
    setup_memory_optimization()
    
    # Create experiment directory structure
    experiment_dir = f"./experiments/{args.experiment_name}"
    lora_output_dir = f"{experiment_dir}/lora_weights"
    
    # Setup accelerator
    accelerator = Accelerator(gradient_accumulation_steps=2)
    device = accelerator.device
    
    print("=" * 60)
    print("         LoRA TRAINING FOR MENSA FOOD GENERATION")
    print("=" * 60)
    print(f"[INIT] Experiment: {args.experiment_name}")
    print(f"[INIT] Dataset: {args.dataset_csv}")
    print(f"[INIT] LoRA weights: {lora_output_dir}")
    print(f"[INIT] Device: {device}")
    
    # Create directories
    os.makedirs(lora_output_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/outputs", exist_ok=True)
    
    print(f"[SETUP] Training setup starts...")

    # Save training configuration
    training_config = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': args.experiment_name,
        'dataset_csv': args.dataset_csv,
        'pretrained_model': args.pretrained_model,
        'resolution': args.resolution,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'device': str(device)
    }
    save_training_config(lora_output_dir, training_config)
    
    # Load models
    print("[MODEL] Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    
    # Freeze models except UNet (LoRA will be applied to UNet)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Configure LoRA
    print(f"[LORA] Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha, 
        # Query, Key, Value, and Output layers infused with LoRA
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,  # Fixed dropout value
        bias="none"
    )
    
    unet.add_adapter(lora_config)
    cast_training_params(unet, dtype=torch.float32)
    
    # Load and prepare dataset
    print("[DATA] Preparing dataset...")
    samples = load_from_csv(args.dataset_csv)
    
    if len(samples) == 0:
        raise ValueError("No valid samples found in dataset!")
    
    dataset = Dataset.from_list(samples)
    dataset = dataset.map(lambda e: tokenize_and_transform(e, tokenizer, args.resolution))
    dataset.set_format(type="torch", columns=["input_ids", "pixel_values"])
    
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch])
        }
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()), 
        lr=args.learning_rate,
        weight_decay=0.01  #  can be tuned (larger values -> Better generalization, smaller values -> May overfit)
    )
    
    # Prepare for training
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    
    print("[SETUP] Training setup complete!")
    print(f"        Dataset size: {len(dataset)}")
    print(f"        Batch size: {args.batch_size}")
    print(f"        Steps per epoch: {len(dataloader)}")
    print(f"        Total epochs: {args.epochs}")
    print(f"        Learning rate: {args.learning_rate}")
    
    # Training loop
    print("-" * 60)
    print(f"[TRAIN] Starting training for {args.epochs} epochs...")
    print("-" * 60)
    unet.train()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # VAE encoding
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, 
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Text encoding
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                
                # UNet prediction
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Loss calculation
                loss = F.mse_loss(model_pred, noise)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping for stability
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Memory cleanup
                if step % 10 == 0:
                    torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"[EPOCH {epoch + 1:2d}] Complete. Average loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"[✓] New best loss! Saving model...")
            accelerator.wait_for_everyone()
            lora_state_dict = get_peft_model_state_dict(unet)
            torch.save(lora_state_dict, os.path.join(lora_output_dir, "pytorch_lora_weights_best.bin"))
        
        # Regular checkpoint saves
        if (epoch + 1) % args.save_steps == 0:
            accelerator.wait_for_everyone()
            lora_state_dict = get_peft_model_state_dict(unet)
            torch.save(lora_state_dict, os.path.join(lora_output_dir, f"pytorch_lora_weights_epoch_{epoch+1}.bin"))
            print(f"[SAVE] Checkpoint saved at epoch {epoch + 1}")
    
    # Final save
    accelerator.wait_for_everyone()
    lora_state_dict = get_peft_model_state_dict(unet)
    torch.save(lora_state_dict, os.path.join(lora_output_dir, "pytorch_lora_weights_final.bin"))
    
    print("-" * 60)
    print("[✓] LoRA fine-tuning training complete!")
    print(f"    Experiment saved to: {experiment_dir}")
    print(f"    LoRA weights saved to: {lora_output_dir}")
    print(f"    Best loss: {best_loss:.4f}")
    print("=" * 60)
    
if __name__ == "__main__":
    main()
