from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionTrainer, StableDiffusionTrainingArguments
from peft import LoraConfig
from datasets import Dataset
import os
import json
from PIL import Image

def load_images_and_captions(dataset_dir):
    with open(os.path.join(dataset_dir, "captions.json")) as f:
        captions = json.load(f)

    examples = []
    for filename, caption in captions.items():
        image_path = os.path.join(dataset_dir, "images", filename)
        if os.path.exists(image_path):
            examples.append({
                "image": Image.open(image_path).convert("RGB"),
                "text": caption
            })
    return examples

def main():
    dataset_dir = "./dataset"
    output_dir = "./lora_cafeteria_output"
    pretrained_model = "CompVis/stable-diffusion-v1-5"

    # Load dataset
    examples = load_images_and_captions(dataset_dir)
    dataset = Dataset.from_list(examples)

    # Training args
    training_args = StableDiffusionTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        max_train_steps=1000,
        save_steps=100,
        image_column="image",
        text_column="text",
        logging_dir="./logs",
        push_to_hub=False,
        resolution=512,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["attn1", "attn2"],
        lora_dropout=0.1,
        bias="none",
    )

    trainer = StableDiffusionTrainer(
        model_name_or_path=pretrained_model,
        train_dataset=dataset,
        args=training_args,
        lora_config=lora_config,
    )

    trainer.train()

if __name__ == "__main__":
    main()
