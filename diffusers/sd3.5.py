import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
).to("cuda")

#transform_type = "vertical"
possible_transform_types = ["vertical", "90degree", "135degree"]

transform_type = "135degree"

assert transform_type in possible_transform_types, f"Transform type must be one of {possible_transform_types}"

style_prompt = "a an oil painting of "
seed = 0

image1,image2 = pipe(
    prompt=f"{style_prompt} a ship",
    prompt_image2=f"{style_prompt} a village in the mountains",
    negative_prompt="",
    num_inference_steps=30,
    height=1024,
    width=1024,
    guidance_scale=4.5,
    transform_type=transform_type,
    time_travel=2,
    generator=torch.Generator("cuda").manual_seed(seed),
    time_travel_range=[20,80],
    denoise_last=True,
    denoise_last_steps=3,
    vis_intermediate=False,
    lwp=True,
)

image1.save("generated_image1.png")
image2.save("generated_image2.png")

