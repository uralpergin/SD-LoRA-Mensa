from diffusers import StableDiffusionPipeline
import torch

# This function loads a fine-tuned Stable Diffusion model with LoRA weights
# and generates an image based on a provided text prompt.
def generate_image(prompt):
    # Load the base Stable Diffusion model (v1-5) with half precision for efficiency
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-5",
        torch_dtype=torch.float16  # Use fp16 for faster inference on GPU
    ).to("cuda")  # Move model to GPU

    # Load the LoRA fine-tuned attention weights
    pipe.unet.load_attn_procs("./lora_cafeteria_output")

    # Generate an image using the given prompt
    image = pipe(prompt).images[0]

    # Save and display the image
    image.save("output.png")
    image.show()

# Example usage: generate an image of a cafeteria tray with specific items
if __name__ == "__main__":
    prompt = "A cafeteria tray with rice, grilled chicken, and broccoli."
    generate_image(prompt)