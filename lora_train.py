"""
Stable Diffusion with LoRA fine-tune Training for Mensa Food Generation
"""

import os
import pandas as pd, os
import torch
from PIL import Image
from datasets import Dataset
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model
from transformers import CLIPTokenizer, CLIPTextModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.training_utils import compute_snr, cast_training_params
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms
from accelerate import Accelerator
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import gc
import itertools
from pathlib import Path


class MensaTorchDataset(TorchDataset):
    def __init__(self, raw_samples, transform, repeat=20):
        """
        Used to get data and transform on the fly to avoid OOM errors.
        raw_samples: list of {"image_path", "input_ids", "negative_input_ids"}
        transform: torchvision transform pipeline
        repeat: how many times to cycle through the base images
        """
        self.raw = raw_samples
        self.transform = transform
        self.repeat = repeat
        self.n = len(raw_samples)

    def __len__(self):
        return self.n * self.repeat

    def __getitem__(self, idx):
        sample = self.raw[idx % self.n]

        img = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.transform(img)

        return {
            "input_ids": sample["input_ids"],
            "negative_input_ids": sample["negative_input_ids"],
            "pixel_values": pixel_values
        }

def get_train_transform(resolution):
    """Returns predefined data augmentation transforms for training"""
    # No augmentation, just resize and normalize
    transform_set = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(0.2,0.2,0.2,0.1)], p=0.3
        ),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return transform_set

def setup_memory_optimization():
    """Setup memory optimization and get current state"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
        torch.backends.cuda.matmul.allow_tf32 = True # Efficent matrix multiplication
        torch.backends.cudnn.allow_tf32 = True # Faster convolutions
        
        # VRAM info
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used_memory = torch.cuda.memory_reserved() / 1024**3
        
        print(f"[INFO] CUDA Device: {device_name}")
        print(f"[INFO] Total VRAM: {total_memory:.1f} GB")
        print(f"[VRAM] Currently used: {used_memory:.2f} GB")

def load_from_csv(csv_path, tokenizer, concept_token="<mensafood>"):
    """
    Returns a list of dicts with:
      - image_path: str
      - input_ids: torch.LongTensor
      - negative_input_ids: torch.LongTensor
    """
    
    df = pd.read_csv(csv_path)
    samples = []
    
    # Standard negative prompt
    neg_prompt = "fork, knife, spoon, napkin, text, watermark, person, hand"
    
    for _, row in df.iterrows():
        path = row["image_path"]
        if not os.path.exists(path):
            continue

        # Build prompts
        pos_prompt = f"{concept_token} {row['description']}"
        
        # Tokenize
        pos_tok = tokenizer(
            pos_prompt, padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.squeeze(0)
        
        neg_tok = tokenizer(
            neg_prompt, padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.squeeze(0)

        samples.append({
            "image_path": path,
            "input_ids": pos_tok,
            "negative_input_ids": neg_tok,
        })

    print(f"Loaded {len(samples)} samples")
    return samples

def save_training_config(output_dir, config):
    """Save training configuration to JSON"""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

def save_lora_and_embedding(output_dir, unet, text_encoder, tokenizer, concept_token, accelerator):
    """Save LoRA weights in diffusers/PEFT format and concept token embedding"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save UNet LoRA attention processors
    unet = accelerator.unwrap_model(unet)
    unet.save_attn_procs(output_path / "unet")
    
    # Save Text Encoder LoRA weights using PEFT method
    text_encoder = accelerator.unwrap_model(text_encoder)
    text_dir = output_path / "text"
    os.makedirs(text_dir, exist_ok=True)
    
    # Get state dict and modify for inference compatibility
    text_encoder_state_dict = get_peft_model_state_dict(text_encoder)
    
    # Add '.default' to weight keys for inference compatibility
    modified_state_dict = {}
    for key, tensor in text_encoder_state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            parts = key.split(".")
            if parts[-1] == "weight":
                new_key = ".".join(parts[:-1]) + ".default.weight"
                modified_state_dict[new_key] = tensor
        else:
            modified_state_dict[key] = tensor
    
    # Save stats for validation during inference
    stats = {}
    for key, tensor in modified_state_dict.items():
        stats[key] = {
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "non_zero": (tensor != 0).sum().item(),
            "shape": list(tensor.shape)
        }
    
    # Save stats and weights
    with open(text_dir / "adapter_stats.json", 'w') as f:
        json.dump(stats, f, default=lambda x: str(x), indent=2)
    
    torch.save(modified_state_dict, text_dir / "adapter_model.bin")
    
    # Save the concept token embedding
    token_id = tokenizer.convert_tokens_to_ids(concept_token)
    embedding_size = text_encoder.text_model.embeddings.token_embedding.weight.shape[0]
    
    if token_id < embedding_size and token_id != tokenizer.unk_token_id:
        token_embedding = text_encoder.text_model.embeddings.token_embedding.weight[token_id].detach().cpu()
        torch.save(token_embedding, output_path / "token_emb.pt")
    
    # Save tokenizer with the new token
    tokenizer.save_pretrained(output_path / "tokenizer")
    
    return output_path

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
    parser.add_argument('--batch_size', type=int, default=6, 
                       help='Training batch size') # CHECK: 6 gives around 9.5 GB VRAM usage
    parser.add_argument('--epochs', type=int, default=15, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=8, # Increase
                       help='LoRA rank - Higher values increase adaptation capacity.')
    parser.add_argument('--lora_alpha', type=int, default=32, 
                       help='LoRA alpha - Higher values increase adaptation strength.')
    parser.add_argument('--save_steps', type=int, default=3, 
                       help='Save model every N epochs')
    parser.add_argument('--concept_token', type=str, default='<mensafood>', 
                       help='Concept token for Mensa food style')
    parser.add_argument('--duplicate', type=int, default=20,
                       help='Number of times to repeat each image in the dataset')

    args = parser.parse_args()

    # Setup memory optimization
    setup_memory_optimization()
    
    # Create experiment directory structure
    experiment_dir = f"./experiments/{args.experiment_name}"
    lora_output_dir = f"{experiment_dir}/lora_weights"
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="fp16" 
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
        'duplicate': args.duplicate,
        'concept_token': args.concept_token,
        'negative_prompt': "fork, knife, spoon, napkin, text, watermark, person, hand", # change if we change negative prompt
        'train_prompt': args.concept_token + " food raw description",
        'scheduler': 'cosine_with_warmup', #change if we change scheduler
        'optimizer': 'adamw', # change if we change optimizer
    }
    save_training_config(lora_output_dir, training_config)
    
    # Load models
    print("[MODEL] Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    
    # Mensa TOKEN setup
    if args.concept_token not in tokenizer.get_vocab():
        print(f"[TOKEN] Adding concept token: {args.concept_token}")
        num_added_tokens = tokenizer.add_tokens([args.concept_token])
        print(f"[TOKEN] Added {num_added_tokens} new tokens")
        # Add padding for safety (to avoid exact boundary issues)
        text_encoder.resize_token_embeddings(len(tokenizer) + 16)
        print(f"[TOKEN] Resized embeddings to {text_encoder.get_input_embeddings().weight.shape[0]}")
    else:
        print(f"[TOKEN] Concept token already exists: {args.concept_token}")
        # Still ensure we have enough padding
        if len(tokenizer) == text_encoder.get_input_embeddings().weight.shape[0]:
            text_encoder.resize_token_embeddings(len(tokenizer) + 16)
            print(f"[TOKEN] Added padding to embeddings, new size: {text_encoder.get_input_embeddings().weight.shape[0]}")
    
    # Print token information for verification
    token_id = tokenizer.convert_tokens_to_ids(args.concept_token)
    print(f"[TOKEN] Concept token '{args.concept_token}' ID: {token_id}")
    print(f"[TOKEN] Tokenizer size: {len(tokenizer)}")
    print(f"[TOKEN] Embedding size: {text_encoder.get_input_embeddings().weight.shape[0]}")
    
    # Ensure the token embedding is trainable
    text_encoder.text_model.embeddings.token_embedding.weight.requires_grad = True
    print("[TOKEN] Token embedding layer set to trainable")
    
    # Enable gradient checkpointing
    unet.enable_gradient_checkpointing()
    
    # Freeze models except UNet and text_encoder
    # TODO: Check if this prevent any learning
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Configure LoRA
    print(f"[LORA] Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha, 
        init_lora_weights="gaussian", # Use Gaussian initialization for LoRA weights
        # Query, Key, Value, and Output layers infused with LoRA
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0, # NOTE: check if need fine-tuning
        bias="none"
    )
    
    unet.add_adapter(lora_config)
    cast_training_params(unet, dtype=torch.float32)
    
    # Add LoRA to text encoder
    print("[LORA] Adding LoRA to text encoder...")
    text_lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","out_proj"],
        lora_dropout=0.1,
        bias="none")
    text_encoder = get_peft_model(text_encoder, text_lora_cfg) # PEFT used

    # Freeze everything that is not LoRA
    # NOTE: looks like add_adapter does this by default, additional checks needed
    print("[FREEZE] Freezing all non-LoRA parameters...")
    for n, p in unet.named_parameters():
        p.requires_grad = "lora_" in n
    for n, p in text_encoder.named_parameters():
        p.requires_grad = "lora_" in n

    text_encoder.text_model.embeddings.token_embedding.weight.requires_grad = True
    
    # Count trainable parameters
    unet_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    text_trainable = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    # print(f"[FREEZE] UNet trainable parameters: {unet_trainable:,}")
    # print(f"[FREEZE] Text encoder trainable parameters: {text_trainable:,}")
    
    # Load and prepare dataset
    print("[DATA] Preparing dataset...")
    
    # Load & tokenize CSV
    raw_samples = load_from_csv(args.dataset_csv, tokenizer, args.concept_token)
    if len(raw_samples) == 0:
        raise ValueError("No valid samples found in dataset!")

    # Build raw samples for PyTorch Dataset
    torch_samples = []
    for sample in raw_samples:
        torch_samples.append({
            "image_path": sample["image_path"],
            "input_ids": sample["input_ids"],
            "negative_input_ids": sample["negative_input_ids"]
        })
    
    # Create PyTorch Dataset with on-the-fly transforms
    train_transform = get_train_transform(args.resolution)
    torch_dataset = MensaTorchDataset(torch_samples, train_transform , repeat=args.duplicate)

    print(f"[DATA] PyTorch Dataset created with {len(torch_dataset)} samples")

    # DataLoader - simplified without multiprocessing
    print("[DATA] Creating DataLoader...")
    dataloader = DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing to avoid crashes
        pin_memory=False,
    )
    print("[OK] DataLoader created successfully")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(
            filter(lambda p: p.requires_grad, unet.parameters()),
            filter(lambda p: p.requires_grad, text_encoder.parameters())
        ), 
        lr=args.learning_rate,
        weight_decay=0.001  # Added weight decay for better regularization
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(dataloader)
    total_training_steps = args.epochs * num_update_steps_per_epoch
    
    # Create a learning rate scheduler with warmup and linear decay
    # Warmup for 10% of the total steps, then linearly decay to 1e-5
    warmup_steps = int(0.20 * total_training_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        num_cycles = 0.5
    )
   


    print(f"[LR] LR Schedule: {args.learning_rate} → 1e-5 with {warmup_steps} warmup steps")
    
    # Prepare for training
    unet, text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, dataloader, lr_scheduler
    )
    
    print("[SETUP] Training setup complete!")
    print(f"        Dataset size: {len(torch_dataset)}")
    print(f"        Batch size: {args.batch_size}")
    print(f"        Steps per epoch: {len(dataloader)}")
    print(f"        Total epochs: {args.epochs}")
    print(f"        Learning rate: {args.learning_rate} → 1e-5")
    print(f"        Warmup steps: {warmup_steps} ({int(0.10*100)}% of training)")
    
    # Training loop
    print("-" * 60)
    print(f"[TRAIN] Starting training for {args.epochs} epochs...")
    print("-" * 60)
    unet.train()
    text_encoder.train()  # Explicitly set text encoder to training mode
    best_loss = float('inf')
    
    # Get token ID for concept token to monitor its embedding
    concept_token_id = tokenizer.convert_tokens_to_ids(args.concept_token)
    print(f"[TRAIN] Monitoring concept token ID: {concept_token_id}")
    
    # Increase learning rate for concept token embedding to emphasize learning
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.learning_rate
        param_group['lr'] = args.learning_rate
    
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
                if torch.rand(1) < 0.3:
                    encoder_hidden_states = text_encoder(batch["negative_input_ids"].to(device))[0]
                else:
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                            
                # UNet prediction
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Loss calculation
                loss = F.mse_loss(model_pred, noise)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping for stability
                if accelerator.sync_gradients:
                    # Clip gradients for both UNet and text encoder
                    params_to_clip = list(unet.parameters()) + list(text_encoder.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Get current learning rate
                current_lr = lr_scheduler.get_last_lr()[0]
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}"
                })
                
                # Memory cleanup
                if step % 10 == 0:
                    torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(dataloader)
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"[EPOCH {epoch + 1:2d}] Complete. Average loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        
        # VRAM report
        if torch.cuda.is_available():
            used_memory = torch.cuda.memory_reserved() / 1024**3
            print(f"[VRAM] Used: {used_memory:.2f} GB")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"[OK] New best loss! Saving model...")
            accelerator.wait_for_everyone()
            save_lora_and_embedding(
                os.path.join(lora_output_dir, "best"),
                unet, text_encoder, tokenizer, args.concept_token, accelerator
            )
        
        # Checkpoint saves
        if (epoch + 1) % args.save_steps == 0:
            accelerator.wait_for_everyone()
            save_lora_and_embedding(
                os.path.join(lora_output_dir, f"epoch_{epoch+1}"),
                unet, text_encoder, tokenizer, args.concept_token, accelerator
            )
            print(f"[SAVE] Checkpoint saved at epoch {epoch + 1}")
    
    # Final save
    accelerator.wait_for_everyone()
    save_lora_and_embedding(
        os.path.join(lora_output_dir, "final"),
        unet, text_encoder, tokenizer, args.concept_token, accelerator
    )
    
    print("-" * 60)
    print("[OK] LoRA fine-tuning training complete!")
    print(f"    Experiment saved to: {experiment_dir}")
    print(f"    LoRA weights saved to: {lora_output_dir}")
    print(f"    Best loss: {best_loss:.4f}")
    print("=" * 60)
    
if __name__ == "__main__":
    main()
