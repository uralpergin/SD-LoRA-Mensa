import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def generate_image(prompt, lora_weights_path, output_path="./outputs/output.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base pipeline
    base_model = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    ).to(device)

    # Load LoRA weights into the UNet
    pipe.unet.load_attn_procs(lora_weights_path)

    # Generate image from prompt
    with torch.autocast("cuda") if torch.cuda.is_available() else torch.no_grad():
        image = pipe(prompt).images[0]

    # Save and show the image
    image.save(output_path)
    image.show()
    print(f"✅ Image saved to: {output_path}")


if __name__ == "__main__":
    prompt = "Apfelstrudel with vanilla sauce and cherry compote, type: Essen 1, diet: Vegetarisch, served at Mensa Institutsviertel"
    lora_weights_path = "./lora_cafeteria_output"  # Adjust if you saved to a different path
    generate_image(prompt, lora_weights_path)
