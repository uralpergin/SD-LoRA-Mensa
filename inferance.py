"""
Inference Script for Mensa Food Generation
"""

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import AutoTokenizer
from PIL import Image
import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from peft import LoraConfig, get_peft_model

def load_pipeline_with_lora(lora_weights_path, concept_token="<mensafood>"):
    """Load and setup the pipeline with LoRA weights"""
    print("[MODEL] Loading pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base pipeline with EulerDiscreteScheduler
    base_model = "CompVis/stable-diffusion-v1-4"
    
    # First load the scheduler separately
    scheduler = EulerDiscreteScheduler.from_pretrained(
        base_model, 
        subfolder="scheduler",
        prediction_type="epsilon"
    )
    
    # Then load the pipeline with the custom scheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        scheduler=scheduler,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        use_peft_backend=True
    ).to(device)
    
    print("[MODEL] Using EulerDiscreteScheduler for better quality")
    
    # Find best weights
    lora_dir = Path(lora_weights_path)
    model_dir = None
    
    # Check best dir first, then final, then any subdirectory with unet
    for check_dir in ["best", "final"]:
        check_path = lora_dir / check_dir
        if check_path.exists() and (check_path / "unet").exists():
            model_dir = check_path
            break
    
    # If still not found, check any subdirectory
    if model_dir is None:
        subdirs = [d for d in lora_dir.iterdir() if d.is_dir() and (d / "unet").exists()]
        if subdirs:
            model_dir = subdirs[0]
        else:
            raise FileNotFoundError(f"No LoRA weights found in {lora_weights_path}")
    
    print(f"[LOAD] Using weights from: {model_dir}")
    
    # Load tokenizer and add concept token
    tokenizer_path = model_dir / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        pipe.tokenizer = tokenizer
    else:
        if concept_token not in pipe.tokenizer.get_vocab():
            pipe.tokenizer.add_tokens([concept_token])
            pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    
    token_id = pipe.tokenizer.convert_tokens_to_ids(concept_token)
    
    # Load concept token embedding
    token_emb_path = model_dir / "token_emb.pt"
    if token_emb_path.exists():
        try:
            emb = torch.load(token_emb_path, map_location=device)
            embedding_size = pipe.text_encoder.text_model.embeddings.token_embedding.weight.shape[0]
            
            # Resize if needed
            if token_id >= embedding_size:
                pipe.text_encoder.resize_token_embeddings(token_id + 16)
            
            # Set token embedding
            with torch.no_grad():
                pipe.text_encoder.text_model.embeddings.token_embedding.weight[token_id] = emb.to(device)
        except Exception as e:
            print(f"[ERROR] Failed to load token embedding: {e}")
    
    # Load LoRA weights
    try:
        # Load UNet LoRA
        unet_lora_path = model_dir / "unet"
        if unet_lora_path.exists():
            pipe.unet.load_attn_procs(str(unet_lora_path))
        
        # Load Text Encoder LoRA
        text_lora_path = model_dir / "text"
        if text_lora_path.exists():
            # Setup PEFT
            peft_cfg = LoraConfig(
                r=8, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                bias="none", lora_dropout=0.0
            )
            pipe.text_encoder = get_peft_model(pipe.text_encoder, peft_cfg)
            
            # Load weights
            peft_state = torch.load(text_lora_path / "adapter_model.bin", map_location=device)
            
            # Fix key format if needed
            fixed_state_dict = {}
            for key, value in peft_state.items():
                if key.endswith(".default.weight"):
                    fixed_state_dict[key] = value
                elif key.endswith(".weight"):
                    fixed_state_dict[key[:-7] + ".default.weight"] = value
                else:
                    fixed_state_dict[key] = value
            
            # Load fixed state dict
            pipe.text_encoder.load_state_dict(fixed_state_dict, strict=False)
        
        # Enable optimizations
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception as e:
        print(f"[ERROR] Error loading LoRA weights: {e}")
        raise
    
    # Store negative prompt for later use
    negative_prompt = "fork, knife, spoon, napkin, text, watermark, person, hand"
    pipe._negative_prompt = negative_prompt
    
    return pipe, str(model_dir)

def generate_image(pipe, food_description, weights_file, output_path, num_inference_steps=50, guidance_scale=7.5, seed=None, concept_token="<mensafood>"):
    """Generate image from prompt using pre-loaded pipeline with EulerDiscreteScheduler"""
    if pipe is None:
        print("[ERROR] Pipeline failed to initialize.")
        return None
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Prepare prompt with concept token
    prompt = f"{concept_token} {food_description} on a white plate, centered"
    print(f"[GEN] Generating: {prompt}")
    
    # Get negative prompt
    negative_prompt = getattr(pipe, '_negative_prompt', 
                            "fork, knife, spoon, napkin, text, watermark, person, hand")
    
    # Generate image
    try:
        with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            ).images[0]
        
        # Save image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"[✓] Image saved: {output_path}")
        
        # Save basic metadata
        metadata = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'guidance_scale': guidance_scale,
            'seed': seed
        }
        
        metadata_path = output_path.replace('.png', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return image
        
    except Exception as e:
        print(f"[ERROR] Failed to generate image: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate Mensa food images using LoRA')
    parser.add_argument('--prompt', required=True, help='Text prompt for image generation')
    parser.add_argument('--experiment_name', default="experiment_default",
                       help='Experiment name')
    parser.add_argument('--steps', type=int, default=50, 
                       help='Number of inference steps')
    parser.add_argument('--guidance', type=float, default=3.5, 
                       help='Guidance scale (higher=follow prompt more closely)')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducibility')
    parser.add_argument('--num_images', type=int, default=1, 
                       help='Number of images to generate')
    parser.add_argument('--concept_token', type=str, default='<mensafood>', 
                       help='Concept token for Mensa food style')
    
    args = parser.parse_args()
    
    # Setup paths
    experiment_dir = f"./experiments/{args.experiment_name}"
    lora_weights_path = f"{experiment_dir}/lora_weights"
    os.makedirs(f"{experiment_dir}/outputs", exist_ok=True)
    
    root_generated_dir = "./generated_images"
    food_folder = "".join(c for c in args.prompt if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
    food_generated_dir = os.path.join(root_generated_dir, food_folder)
    os.makedirs(food_generated_dir, exist_ok=True)

    root_original_dir = "./fid_original_images"
    original_folder = "".join(c for c in args.prompt if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
    original_generated_dir = os.path.join(root_original_dir, original_folder)
    os.makedirs(original_generated_dir, exist_ok=True)

    if not os.path.exists(lora_weights_path):
        print(f"[ERROR] Experiment not found: {args.experiment_name}")
        return 1
    
    # Create output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c for c in args.prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
    output_path = f"{experiment_dir}/outputs/{safe_prompt}_{timestamp}.png"
    
    # Load pipeline
    try:
        pipe, weights_file = load_pipeline_with_lora(lora_weights_path, args.concept_token)
    except Exception as e:
        print(f"[ERROR] Failed to load pipeline: {e}")
        return 1
    
    # Generate images
    for i in range(args.num_images):
        current_output = output_path.replace('.png', f'_{i+1}.png') if args.num_images > 1 else output_path
        current_seed = args.seed + i if args.seed is not None and args.num_images > 1 else args.seed
        
        image = generate_image(
            pipe=pipe,
            food_description=args.prompt,
            weights_file=weights_file,
            output_path=current_output,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=current_seed,
            concept_token=args.concept_token
        )

        if image is not None:
            root_image_path = os.path.join(food_generated_dir, f"{args.prompt}.png")
            image.save(root_image_path)
            print(f"[✓] Image also saved to: {root_image_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
