"""
Inference Script for Mensa Food Generation
"""

import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer
from PIL import Image
import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from peft import LoraConfig, get_peft_model

def load_pipeline_with_lora(lora_weights_path, concept_token="<mensafood>"):
    """Load and setup the pipeline with LoRA weights once"""
    # TODO: this function needs to be cleaning and commenting
    print("[MODEL] Loading pipeline with LoRA weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load base pipeline
    base_model = "CompVis/stable-diffusion-v1-4"
    print(f"[MODEL] Loading base model: {base_model}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Determine the LoRA directory to load from
    lora_dir = Path(lora_weights_path)
    
    # Look for the best/final LoRA weights in the directory structure
    best_dir = lora_dir / "best"
    final_dir = lora_dir / "final"
    
    if best_dir.exists():
        model_dir = best_dir
        print("[LORA] Using best model weights")
    elif final_dir.exists():
        model_dir = final_dir
        print("[LORA] Using final model weights")
    else:
        # Try to find any subdirectory with LoRA weights
        subdirs = [d for d in lora_dir.iterdir() if d.is_dir() and (d / "unet").exists()]
        if subdirs:
            model_dir = subdirs[0]  # Use the first available
            print(f"[LORA] Using model weights from: {model_dir.name}")
        else:
            raise FileNotFoundError(f"No LoRA weights found in {lora_weights_path}")
    
    print(f"[LOAD] Loading LoRA weights from: {model_dir}")
    
    # Load tokenizer first and re-register the placeholder token
    tokenizer_path = model_dir / "tokenizer"
    if tokenizer_path.exists():
        print("[TOKEN] Loading saved tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        pipe.tokenizer = tokenizer  # Keep pipe in sync
        token_id = tokenizer.convert_tokens_to_ids(concept_token)
        print(f"[TOKEN] Concept token '{concept_token}' ID from saved tokenizer: {token_id}")
    else:
        # Fallback: add token to existing tokenizer
        print("[TOKEN] Adding concept token to existing tokenizer...")
        if concept_token not in pipe.tokenizer.get_vocab():
            pipe.tokenizer.add_tokens([concept_token])
            pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        token_id = pipe.tokenizer.convert_tokens_to_ids(concept_token)
        print(f"[TOKEN] Concept token '{concept_token}' ID: {token_id}")

    # Restore concept token embedding
    token_emb_path = model_dir / "token_emb.pt"
    if token_emb_path.exists():
        print("[TOKEN] Loading concept token embedding...")
        emb = torch.load(token_emb_path, map_location=device)
        
        # Safety check: ensure token_id is within bounds
        embedding_size = pipe.text_encoder.text_model.embeddings.token_embedding.weight.shape[0]
        if token_id >= embedding_size:
            print(f"[WARN] Token ID {token_id} is out of bounds for embedding size {embedding_size}, resizing embeddings")
            # Add padding of 16 tokens for safety (to avoid exact boundary issues)
            pipe.text_encoder.resize_token_embeddings(token_id + 16)
            print(f"[TOKEN] Embeddings resized to {pipe.text_encoder.text_model.embeddings.token_embedding.weight.shape[0]}")
        
        # Set the concept token embedding with proper error handling
        try:
            with torch.no_grad():
                pipe.text_encoder.text_model.embeddings.token_embedding.weight[token_id] = emb.to(device)
            print(f"[TOKEN] ✓ Concept token embedding successfully restored for token ID: {token_id}")
        except Exception as e:
            print(f"[ERROR] Failed to set token embedding: {e}")
            return None, None
    else:
        print(f"[WARN] No concept token embedding found at {token_emb_path}")

    # Load LoRA weights for UNet and CLIP 
    try:
        unet_lora_path = model_dir / "unet"
        text_lora_path = model_dir / "text"
        
        if unet_lora_path.exists():
            print("[LORA] Loading UNet LoRA weights...")
            pipe.unet.load_attn_procs(str(unet_lora_path))
            print("[LORA] UNet LoRA weights loaded successfully")
        else:
            print(f"[WARN] UNet LoRA weights not found at {unet_lora_path}")
        
        if text_lora_path.exists():
            print("[LORA] Loading Text Encoder LoRA weights (PEFT format)...")
            # Use PEFT to load text encoder LoRA weights
            from peft import LoraConfig, get_peft_model
            
            peft_cfg = LoraConfig(
                r=8, lora_alpha=32,
                target_modules=["q_proj","k_proj","v_proj","out_proj"],
                bias="none", lora_dropout=0.0  # No dropout for inference
            )
            pipe.text_encoder = get_peft_model(pipe.text_encoder, peft_cfg)
            
            # Load the saved PEFT state with error handling
            try:
                peft_state = torch.load(text_lora_path / "adapter_model.bin", map_location=device)
                
                # Debug: Print state dict keys to verify alignment
                print("[DEBUG] First 5 keys in saved state dict:")
                for idx, key in enumerate(list(peft_state.keys())[:5]):
                    print(f"  - {key}")
                
                # Debug: Print model keys to compare with state dict
                print("[DEBUG] First 5 LoRA keys in model:")
                lora_keys = [name for name, _ in pipe.text_encoder.named_parameters() if "lora_" in name]
                for idx, key in enumerate(lora_keys[:5]):
                    print(f"  - {key}")
                
                # Calculate key matching stats
                matching_keys = 0
                missing_keys = []
                for key in peft_state.keys():
                    if key in dict(pipe.text_encoder.named_parameters()):
                        matching_keys += 1
                    else:
                        missing_keys.append(key)
                
                print(f"[DEBUG] Keys match statistics: {matching_keys}/{len(peft_state)} keys matched")
                if missing_keys:
                    print(f"[DEBUG] First 3 missing keys: {missing_keys[:3]}")
                    
                # Load state dict with strict=False to avoid errors
                pipe.text_encoder.load_state_dict(peft_state, strict=False)
                
                # Verify weights were actually loaded by checking a few parameters
                print("[DEBUG] Verifying weights were loaded:")
                sample_loaded = False
                for name, param in pipe.text_encoder.named_parameters():
                    if "lora_" in name and param.requires_grad:
                        print(f"  - {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                        print(f"    non-zero values: {(param.data != 0).sum().item()}/{param.data.numel()}")
                        sample_loaded = True
                        break  # Just show one sample
                
                # Load and compare with saved stats if available
                stats_path = text_lora_path / "adapter_stats.json"
                if stats_path.exists():
                    try:
                        with open(stats_path, 'r') as f:
                            saved_stats = json.load(f)
                        print("[DEBUG] Comparing with saved stats from training:")
                        matches = 0
                        for name, param in pipe.text_encoder.named_parameters():
                            if "lora_" in name and name in saved_stats:
                                saved_mean = saved_stats[name]["mean"]
                                current_mean = param.data.mean().item()
                                saved_nonzero = saved_stats[name]["non_zero"]
                                current_nonzero = (param.data != 0).sum().item()
                                
                                mean_diff = abs(float(saved_mean) - current_mean)
                                nonzero_diff = abs(int(saved_nonzero) - current_nonzero)
                                
                                if mean_diff < 1e-5 and nonzero_diff == 0:
                                    matches += 1
                                
                                if matches < 3:  # Only show first few for brevity
                                    print(f"  - {name}: mean diff={mean_diff:.8f}, nonzero diff={nonzero_diff}")
                        
                        print(f"[DEBUG] Stats match verification: {matches}/{len([n for n, _ in pipe.text_encoder.named_parameters() if 'lora_' in n])}")
                    except Exception as e:
                        print(f"[DEBUG] Error comparing stats: {e}")
                
                if not sample_loaded:
                    print("[WARN] Could not find any LoRA parameters in the model!")
                
                print("[LORA] ✓ Text Encoder LoRA weights loaded successfully (PEFT)")
            except Exception as e:
                print(f"[ERROR] Failed to load Text Encoder LoRA weights: {e}")
                print("[WARN] Continuing without Text Encoder LoRA weights")
        else:
            print(f"[WARN] Text encoder LoRA weights not found at {text_lora_path}")
        
        # Don't fuse LoRA for PEFT text encoder - only apply to UNet if needed
        try:
            # Note: PEFT text encoder doesn't need fusing, UNet LoRA is already loaded
            print("[LORA] LoRA weights loaded successfully")
        except Exception as e:
            print(f"[WARN] Could not optimize LoRA weights: {e}")
        
    except Exception as e:
        print(f"[ERROR] Error loading LoRA weights: {e}")
        raise e  # Re-raise the error instead of falling back to legacy
    
    # Safety check
    try:
        if hasattr(pipe.unet, 'mid_block') and hasattr(pipe.unet.mid_block, 'attentions'):
            attn_proc = pipe.unet.mid_block.attentions[0].processor
            if "Lora" in attn_proc.__class__.__name__:
                print("[✓] LoRA successfully applied to UNet")
            else:
                print(f"[WARN] UNet attention processor: {attn_proc.__class__.__name__}")
    except Exception as e:
        print(f"[INFO] Could not verify LoRA application: {e}")
    
    # Enable memory efficient attention if available
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        print("[OPT] Memory optimizations enabled")
    except:
        print("[!] Memory optimizations not available")
    
    # Store negative prompt for later use
    negative_prompt = "fork, knife, spoon, napkin, text, watermark, person, hand"
    pipe._negative_prompt = negative_prompt
    print(f"[CONFIG] Negative prompt stored: {negative_prompt}")
    
    return pipe, str(model_dir)

def generate_image(pipe, food_description, weights_file, output_path, num_inference_steps=50, guidance_scale=7.5, seed=None, concept_token="<mensafood>"):
    """Generate image from prompt using pre-loaded pipeline"""
    if pipe is None:
        print("[ERROR] Pipeline failed to initialize. Cannot generate image.")
        return None
        
    print("[GEN] Image Generation Starts")
    device = pipe.device
    
    # Use seed if exists
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        print(f"[SEED] Using seed: {seed}")
    
    # Prepare prompt with concept token
    base_prompt = "{concept_token} {food_description} on a white plate, placed on a grey tray"
    prompt = base_prompt.format(concept_token=concept_token, food_description=food_description)
    
    print(f"[PROMPT] Positive: {prompt}")
    print(f"[GEN] Generating image...")
    
    # Generate image with stored negative prompt
    negative_prompt = getattr(pipe, '_negative_prompt', "fork, knife, spoon, napkin, text, watermark, person, hand")
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
    print(f"[✓] Image saved to: {output_path}")
    
    # Save generation metadata
    negative_prompt = getattr(pipe, '_negative_prompt', "fork, knife, spoon, napkin, text, watermark, person, hand")
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'prompt': prompt,
        'concept_token': concept_token,
        'negative_prompt': negative_prompt,
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
                       help='Number of inference steps (using default 50 for best quality)')
    # Guidance scale:
    # Larger values force model to follow prompt more closely (unnatural images)
    # Lower values allow more creativity but may diverge from prompt (more natural images)
    parser.add_argument('--guidance', type=float, default=7.5, 
                       help='Guidance scale')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducibility')
    parser.add_argument('--num_images', type=int, default=1, 
                       help='Number of images to generate')
    parser.add_argument('--concept_token', type=str, default='<mensafood>', 
                       help='Concept token for Mensa food style')
    
    args = parser.parse_args()
    
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
    
    # Load pipeline with LoRA weights once outside the loop
    try:
        pipe, weights_file = load_pipeline_with_lora(lora_weights_path, args.concept_token)
        print("[✓] Pipeline loaded successfully")
    except Exception as e:
        print(f"[x] Error loading pipeline: {e}")
        return 1
    
    # Generate multiple images
    # As default, generate 1 image
    # TODO: never tried multiple images so need to test
    for i in range(args.num_images):
        if args.num_images > 1:
            current_output = output_path.replace('.png', f'_{i+1}.png')
            current_seed = args.seed + i if args.seed is not None else None
        else:
            current_output = output_path
            current_seed = args.seed
        
        try:
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
