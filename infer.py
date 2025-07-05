from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.unet.load_attn_procs("./lora_cafeteria_output")

    image = pipe(prompt).images[0]
    image.save("output.png")
    image.show()

if __name__ == "__main__":
    prompt = "A cafeteria tray with rice, grilled chicken, and broccoli."
    generate_image(prompt)