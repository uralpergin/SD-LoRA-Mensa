"""
This script generates images using only the base Stable Diffusion model without LoRA.
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
import os
import json
from datetime import datetime

def generate_image(food_description, output_path, num_inference_steps=50, guidance_scale=7.5, seed=None):
    """Generate image from prompt using vanilla Stable Diffusion without LoRA"""
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
    
    print("[INFO] Using vanilla Stable Diffusion without LoRA")
    
    # Enable memory efficient attention if available
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        print("[OPT] Memory optimizations enabled")
    except:
        print("[!] Memory optimizations not available")
    
    # Prepare prompt
    base_prompt = "{food_description} on a white plate, placed on a grey tray"
    prompt = base_prompt.format(food_description=food_description)
    
    # Negative prompt same as training
    negative_prompt = "fork, knife, spoon, napkin, text, watermark, person, hand"
    
    print(f"[PROMPT] {prompt}")

    print(f"[GEN] Generating image...")
    
    # Generate image with negative prompt
    with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
        image = pipe(
            prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512
        ).images[0]
    
    # VRAM info
    if torch.cuda.is_available():
        used_memory = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] Used: {used_memory:.2f} GB")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    image.save(output_path)
    print(f"[OK] Image saved to: {output_path}")
    
    # Save generation metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'model': 'CompVis/stable-diffusion-v1-4 (without LoRA)',
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
    parser = argparse.ArgumentParser(description='Generate Mensa food images using vanilla Stable Diffusion')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--steps', type=int, default=50, 
                       help='Number of inference steps (using default 50 for best quality)')
    parser.add_argument('--guidance', type=float, default=7.5, 
                       help='Guidance scale')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducibility')
    parser.add_argument('--num_images', type=int, default=1, 
                       help='Number of images to generate')
    parser.add_argument('--experiment_name', type=str, default="vanilla_sd", 
                       help='Name of dir with results.')
    
    args = parser.parse_args()
    
    # Use a dedicated directory for vanilla SD results
    experiment_dir = f"./experiments/{args.experiment_name}"
    
    # Output directory for generated images
    os.makedirs(f"{experiment_dir}/outputs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c for c in args.prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')
    output_path = f"{experiment_dir}/outputs/{safe_prompt}_{timestamp}.png"
    
    print("=" * 60)
    print("         MENSA FOOD IMAGE GENERATION (VANILLA SD)")
    print("=" * 60)
    print(f"[INIT] Model: CompVis/stable-diffusion-v1-4 (without LoRA)")
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
                food_description=args.prompt,
                output_path=current_output,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=current_seed
            )
            
            print(f"[OK] Generated image {i+1}/{args.num_images}")
            
        except Exception as e:
            print(f"[x] Error generating image: {e}")
            return 1
    
    print("-" * 60)
    print("[OK] Mensa Food Image Generation Completed")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())
