"""
Inference Script for Mensa Food Generation

This script generates images using the fine-tuned LoRA model.
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
import os
import json
from datetime import datetime

def generate_image(prompt, lora_weights_path, output_path, num_inference_steps=50, guidance_scale=7.5, seed=None):
    """Generate image from prompt using LoRA weights"""
    print("[GEN] Image Generation Starts")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"[SEED] Using seed: {seed}")
    
    # Load base pipeline
    base_model = "CompVis/stable-diffusion-v1-4"
    print(f"[MODEL] Loading base model: {base_model}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Load LoRA weights
    if os.path.isfile(lora_weights_path):
        # Single file
        weights_file = lora_weights_path
    elif os.path.isdir(lora_weights_path):
        # Directory
        best_weights = os.path.join(lora_weights_path, "pytorch_lora_weights_best.bin")
        final_weights = os.path.join(lora_weights_path, "pytorch_lora_weights_final.bin")
        regular_weights = os.path.join(lora_weights_path, "pytorch_lora_weights.bin")
        
        if os.path.exists(best_weights):
            weights_file = best_weights
            print("[LORA] Using best model weights")
        elif os.path.exists(final_weights):
            weights_file = final_weights
            print("[LORA] Using final model weights")
        elif os.path.exists(regular_weights):
            weights_file = regular_weights
            print("[LORA] Using regular model weights")
        else:
            raise FileNotFoundError(f"No LoRA weights found in {lora_weights_path}")
    else:
        raise FileNotFoundError(f"LoRA weights not found: {lora_weights_path}")
    
    print(f"[LOAD] Loading LoRA weights: {weights_file}")
    
    # Load LoRA weights
    try:
        pipe.load_lora_weights(os.path.dirname(weights_file), weight_name=os.path.basename(weights_file))
        print("[✓] LoRA weights loaded (new method)")
    except Exception as e1:
        try:
            pipe.unet.load_attn_procs(weights_file)
            print("[✓] LoRA weights loaded (old method)")
        except Exception as e2:
            print(f"[x] Error loading LoRA weights:")
            print(f"    Method 1 error: {e1}")
            print(f"    Method 2 error: {e2}")
            raise RuntimeError("Cannot load LoRA weights...")
    
    # Enable memory efficient attention if available
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        print("[OPT] Memory optimizations enabled")
    except:
        print("[!] Memory optimizations not available")
    
    print(f"[PROMPT] {prompt}")
    print(f"[GEN] Generating image...")
    
    # Generate image
    with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
        image = pipe(
            prompt, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512
        ).images[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    image.save(output_path)
    print(f"[✓] Image saved to: {output_path}")
    
    # Save generation metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'prompt': prompt,
        'lora_weights': weights_file,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'seed': seed,
        'output_path': output_path
    }
    
    metadata_path = output_path.replace('.png', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Generate Mensa food images using LoRA')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--experiment_name', default="experiment_default",
                       help='Experiment name (default: experiment_default)')
    parser.add_argument('--steps', type=int, default=50, 
                       help='Number of inference steps')
    # Guidance scale:
    # Larger values force model to follow prompt more closely (unnatural images)
    # Lower values allow more creativity but may diverge from prompt (more natural images)
    parser.add_argument('--guidance', type=float, default=7.5, 
                       help='Guidance scale')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducibility')
    parser.add_argument('--num_images', type=int, default=1, 
                       help='Number of images to generate')
    
    args = parser.parse_args()
    
    # Record the results of inference to created experiment directory during training
    experiment_dir = f"./experiments/{args.experiment_name}"
    lora_weights_path = f"{experiment_dir}/lora_weights"
    
    if not os.path.exists(lora_weights_path):
        print(f"[x] Experiment not found: {args.experiment_name}")
        print(f"    Expected path: {lora_weights_path}")
        return 1
    
    # Output directory for generated images
    os.makedirs(f"{experiment_dir}/outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Added if we want to run experiment with different prompts or seeds
    safe_prompt = "".join(c for c in args.prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')
    output_path = f"{experiment_dir}/outputs/{safe_prompt}_{timestamp}.png"
    
    print("=" * 60)
    print("         MENSA FOOD IMAGE GENERATION")
    print("=" * 60)
    print(f"[INIT] Experiment: {args.experiment_name}")
    print(f"[INIT] LoRA weights: {lora_weights_path}")
    print(f"[INIT] Output: {output_path}")
    print("-" * 60)
    
    # Generate multiple images (should be given with --num_images flag)
    # As default, generate 1 image
    for i in range(args.num_images):
        if args.num_images > 1:
            current_output = output_path.replace('.png', f'_{i+1}.png')
            current_seed = args.seed + i if args.seed is not None else None
        else:
            current_output = output_path
            current_seed = args.seed
        
        try:
            image = generate_image(
                prompt=args.prompt,
                lora_weights_path=lora_weights_path,
                output_path=current_output,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=current_seed
            )
            
            if args.num_images > 1:
                print(f"[✓] Generated image {i+1}/{args.num_images}")
            
        except Exception as e:
            print(f"[x] Error generating image: {e}")
            return 1
    
    print("-" * 60)
    print("[✓] Mensa Food Image Generation Completed")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())
