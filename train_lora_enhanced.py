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
        
        # VRAM info
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used_memory = torch.cuda.memory_reserved() / 1024**3
        
        print(f"[INFO] CUDA Device: {device_name}")
        print(f"[INFO] Total VRAM: {total_memory:.1f} GB")
        print(f"[VRAM] Currently used: {used_memory:.2f} GB")

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
        
        # Handle relative paths in the dataset with a simpler approach
        # All paths in the CSV are now stored as relative paths from mensa-lora root
        # We just need to check if the path exists or needs the prefix
        if not os.path.exists(image_path):
            # Try with project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets mensa-lora directory
            full_path = os.path.join(current_dir, image_path)
            if os.path.exists(full_path):
                image_path = full_path
        
        # Final check if image exists
        if not os.path.exists(image_path):
            print(f"[x] Image not found: {image_path}")
            print(f"    Tried absolute path: {image_path}")
            print(f"    Tried with project root: {full_path}")
            missing_count += 1
            continue
        
        description = row["description"]
        
        # Generic prompt template
        # TODO: Side dish case can be seperated based on side dish and the word can be given as variable
        base_prompt = "A mensa meal, {food_description} served on a white plate, placed on a grey tray, realistic photo, centered, high-angle view, outdoors, bright lighting, wide angle, high resolution"
        
        prompt = base_prompt.format(food_description=description)
        #prompt = description
        # Define negative prompt (what we DON'T want)
        negative_prompt = "blurry, low quality, distorted, bad lighting, dark, grainy, pixelated, deformed, messy, scattered, utensils, cutlery, fork, knife, spoon, napkin, hand, person, text, out of frame, unappetizing, overexposed, underexposed, noise, artifacts, watermark"
        #negative_prompt = ""
        #print(f"Positive: {prompt}")
        #print(f"Negative: {negative_prompt}")
        
        try:
            print(f"[LOAD] Attempting to load: {image_path}")
            image = Image.open(image_path).convert("RGB")
            samples.append({
                "image": image, 
                "label": prompt,
                "negative_label": negative_prompt
            })
        except Exception as e:
            print(f"[x] Failed to load image {image_path}: {e}")
            missing_count += 1

    print(f"[OK] Loaded {len(samples)} images successfully")
    if missing_count > 0:
        print(f"[!] {missing_count} images were missing or failed to load")
    
    return samples

def tokenize_and_transform(sample, tokenizer, resolution):
    """
    Tokenize text and transform images with both positive and negative prompts
    sample: Dictionary with 'image', 'label', and 'negative_label' keys
    tokenizer: CLIPTokenizer instance
    resolution: Image resolution for training
    """
    # Tokenize positive prompt (what we want)
    positive_encoding = tokenizer(
        sample["label"], # meal label 
        padding="max_length", # pad each text to max length
        truncation=True, # if the text is to long cut off
        max_length=tokenizer.model_max_length, 
        return_tensors="pt", # return PyTorch tensors
    )
    
    # Tokenize negative prompt (what we don't want)
    negative_encoding = tokenizer(
        sample["negative_label"], # negative prompts
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    
    sample["input_ids"] = positive_encoding["input_ids"].squeeze(0) # Positive tokenized text
    sample["negative_input_ids"] = negative_encoding["input_ids"].squeeze(0) # Negative tokenized text
    
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
    parser.add_argument('--batch_size', type=int, default=2, 
                       help='Training batch size') # Increased to 2 per supervisor suggestion
    parser.add_argument('--epochs', type=int, default=15, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, 
                       help='Learning rate - increased for stronger adaptation')
    # LoRA Rank:
    # Larger values allow more adaptation (capacity of the adaptation increase) but require more VRAM
    parser.add_argument('--lora_r', type=int, default=32, 
                       help='LoRA rank - increased from 16 to 32 for much stronger adaptation') 
    # LoRA Alpha:
    # Larger values increase adaptation strength (output affects from tuning more) but require more VRAM
    parser.add_argument('--lora_alpha', type=int, default=128, 
                       help='LoRA alpha - increased from 64 to 128 for maximum effect')
    parser.add_argument('--save_steps', type=int, default=3, 
                       help='Save model every N epochs')
    parser.add_argument('--cfg_weight', type=float, default=0.3, 
                       help='Classifier-Free Guidance weight (0.0 = no CFG, 0.1 = light CFG, 0.3 = strong CFG)')
    
    args = parser.parse_args()
    
    # Setup memory optimization
    setup_memory_optimization()
    
    # Create experiment directory structure
    experiment_dir = f"./experiments/{args.experiment_name}"
    lora_output_dir = f"{experiment_dir}/lora_weights"
    
    # Setup accelerator with mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=4,  # Increased from 2 to 4 to handle larger batch size
        mixed_precision="fp16"  # Use mixed precision for memory efficiency
    )
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
        'cfg_weight': args.cfg_weight,
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
    
    # Enable gradient checkpointing for memory efficiency
    print("[MEMORY] Enabling gradient checkpointing...")
    unet.enable_gradient_checkpointing()
    
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
    dataset.set_format(type="torch", columns=["input_ids", "negative_input_ids", "pixel_values"])
    
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "negative_input_ids": torch.stack([x["negative_input_ids"] for x in batch]),
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
    print(f"        CFG weight: {args.cfg_weight}")
    if args.cfg_weight > 0:
        print(f"        Negative prompts: ENABLED")
    else:
        print(f"        Negative prompts: DISABLED")
    
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
                
                # Text encoding with CFG support
                with torch.no_grad():
                    # Positive prompt embeddings (what we want)
                    positive_embeddings = text_encoder(batch["input_ids"].to(device))[0]
                    
                    if args.cfg_weight > 0:
                        # Negative prompt embeddings (what we don't want)
                        negative_embeddings = text_encoder(batch["negative_input_ids"].to(device))[0]
                        
                        # Combine embeddings with CFG weighting
                        # CFG formula: positive + cfg_weight * (positive - negative)
                        encoder_hidden_states = positive_embeddings + args.cfg_weight * (positive_embeddings - negative_embeddings)
                    else:
                        # No CFG, use only positive embeddings
                        encoder_hidden_states = positive_embeddings
                
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
        
        # Simple VRAM usage at end of epoch
        if torch.cuda.is_available():
            used_memory = torch.cuda.memory_reserved() / 1024**3
            print(f"[VRAM] Used: {used_memory:.2f} GB")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"[OK] New best loss! Saving model...")
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
    print("[OK] LoRA fine-tuning training complete!")
    print(f"    Experiment saved to: {experiment_dir}")
    print(f"    LoRA weights saved to: {lora_output_dir}")
    print(f"    Best loss: {best_loss:.4f}")
    print("=" * 60)
    
if __name__ == "__main__":
    main()
