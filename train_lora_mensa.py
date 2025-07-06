from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionTrainer, StableDiffusionTrainingArguments
from peft import LoraConfig
from datasets import Dataset
from PIL import Image
import pandas as pd
import os

#THE ORIGINAL LORA TRAINING SCRIPT FOR POKEMON DATASET SAYAKPAUL:
# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
# It is much more complex than this code, but it is a good reference for the future.

# Example csv reading :
# Given a CSV file with the following structure: 
# Currywurst oder planted Currywurst Pommes frites,no,no,Wochenangebot,Vegan auf Anfrage,Mensa Rempartstrae,/work/dlclarge2/matusd-dl_lab_project/images/task_2/2025-01-22/mensa-rempartstrae_wochenangebot.jpg
# Generated prompt will look like: 
# Currywurst oder planted Currywurst Pommes frites, type: Wochenangebot, diet: Vegan auf Anfrage, served at Mensa Rempartstrae

def load_from_csv(csv_path):
    df = pd.read_csv(csv_path)  # Load the CSV file into a DataFrame

    examples = []  # Will hold image-text pairs

    for _, row in df.iterrows():
        image_path = row["image_path"]  # Use absolute image path directly
        if not os.path.exists(image_path):
            continue  # Skip if image doesn't exist

        # Construct the prompt from various metadata fields
        parts = [
            str(row.get("description", "")).strip(),
            "with side salad" if str(row.get("Beilagensalat", "")).strip().lower() in ["ja", "yes"] else "",
            "with regional apple" if str(row.get("Regio Apfel", "")).strip().lower() in ["ja", "yes"] else "",
            f"type: {row.get('type', '').strip()}",
            f"diet: {row.get('diet', '').strip()}",
            f"served at {row.get('mensa', '').strip()}"
        ]

        # Join non-empty components into a final prompt
        prompt = ", ".join([p for p in parts if p])

        # Append the example to our dataset list
        examples.append({
            "image": Image.open(image_path).convert("RGB"),
            "text": prompt
        })

    return examples

def main():
    # Set paths for the dataset and model output
    dataset_csv = "./dataset/metadata.csv"  # CSV with metadata and paths
    output_dir = "./lora_cafeteria_output"  # Where to save LoRA weights
    pretrained_model = "CompVis/stable-diffusion-v1-5"  # Pretrained SD base

    # Load the examples using our CSV-based loader
    examples = load_from_csv(dataset_csv)
    dataset = Dataset.from_list(examples)  # Convert to HuggingFace Dataset

    # Define training arguments
    training_args = StableDiffusionTrainingArguments(
        output_dir=output_dir,  # Save path for model checkpoints
        per_device_train_batch_size=1,  # Low batch size for limited VRAM
        learning_rate=1e-4,
        max_train_steps=1000,  # Total training steps
        save_steps=100,  # Save every N steps
        image_column="image",  # Column with image data
        text_column="text",  # Column with prompt text
        logging_dir="./logs",  # TensorBoard logs
        push_to_hub=False,  # Do not upload to HuggingFace Hub
        resolution=512,  # Image resolution
    )

    # Define LoRA (Low-Rank Adaptation) configuration
    lora_config = LoraConfig(
        r=4,  # Rank (dimensionality of LoRA matrices)
        lora_alpha=16,
        target_modules=["attn1", "attn2"],  # Which parts of UNet to adapt
        lora_dropout=0.1,
        bias="none"  # Do not train bias terms
    )

    # Set up the trainer with model, data, args, and LoRA config
    trainer = StableDiffusionTrainer(
        model_name_or_path=pretrained_model,
        train_dataset=dataset,
        args=training_args,
        lora_config=lora_config,
    )

    # Start the fine-tuning process
    trainer.train()

if __name__ == "__main__":
    main()

