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

print(torch.__version__)
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
# Load images and prompts from CSV
def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    examples = []

    for _, row in df.iterrows():
        # Use only the image filename, ignore bad prefix if needed
        image_filename = os.path.basename(row["image_path"])
        image_path = os.path.join("dataset", image_filename)

        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue

        parts = [
            str(row.get("description", "")).strip(),
            "with side salad" if str(row.get("Beilagensalat", "")).lower() in ["yes", "ja"] else "",
            "with regional apple" if str(row.get("Regio Apfel", "")).lower() in ["yes", "ja"] else "",
            f"type: {row.get('type', '')}",
            f"diet: {row.get('diet', '')}",
            f"served at {row.get('mensa', '')}"
        ]

        prompt = ", ".join([p for p in parts if p])
        examples.append({"image": Image.open(image_path).convert("RGB"), "text": prompt})

    return examples


# Tokenizer and transforms
def tokenize_and_transform(example, tokenizer, resolution):
    encoding = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

    # Squeeze to get (seq_len,) shape instead of (1, seq_len)
    example["input_ids"] = encoding["input_ids"].squeeze(0)

    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=Image.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    example["pixel_values"] = transform(example["image"])
    return example


# Main training
def main():
    accelerator = Accelerator()
    device = accelerator.device
    resolution = 512

    # Set paths for the dataset and model output
    dataset_csv = "./dataset/dataset.csv"  # CSV with metadata and paths
    output_dir = "./lora_cafeteria_output"  # Where to save LoRA weights
    pretrained_model = "CompVis/stable-diffusion-v1-4"  # Pretrained SD base

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

    # Freeze all layers except LoRA
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none"
    )

    unet.add_adapter(lora_config)
    cast_training_params(unet, dtype=torch.float32)

    examples = load_from_csv(dataset_csv)
    dataset = Dataset.from_list(examples)
    dataset = dataset.map(lambda e: tokenize_and_transform(e, tokenizer, resolution))
    dataset.set_format(type="torch", columns=["input_ids", "pixel_values"])

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch])
        }

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=1e-4)

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    unet.train()
    for epoch in range(10):
        for step, batch in enumerate(tqdm(dataloader)):
            with accelerator.accumulate(unet):
                # VAE encode
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(model_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        print(f"Epoch {epoch + 1} complete. Loss: {loss.item():.4f}")

    accelerator.wait_for_everyone()
    lora_state_dict = get_peft_model_state_dict(unet)
    torch.save(lora_state_dict, os.path.join(output_dir, "pytorch_lora_weights.bin"))
    print("LoRA fine-tuning complete.")

if __name__ == "__main__":
    main()
