import argparse
import os
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
).to("cuda")

#transform_type = "vertical"
possible_transform_types = ["vertical", "horizontal", "90rot", "135rot", "180rot", "jigsaw"]

parser = argparse.ArgumentParser(description="Generate SD3.5 dual prompts with a shared style.")
parser.add_argument("--style-prompt", default="a pop art of ", help="Style prefix applied to both prompts.")
parser.add_argument("--prompt1", default="albert einstein", help="First subject prompt.")
parser.add_argument("--prompt2", default="marilyn monroe", help="Second subject prompt.")
parser.add_argument("--output-dir", default=".", help="Directory to save generated images.")
parser.add_argument(
    "--transform",
    default="jigsaw",
    choices=possible_transform_types,
    help="Anamorphosis transform type to apply.",
)
parser.add_argument("--seed", type=int, default=1, help="Random seed for generation.")
args = parser.parse_args()

style_prompt = args.style_prompt
transform_type = args.transform
seed = args.seed

image1, image2 = pipe(
    prompt=f"{style_prompt} {args.prompt1}",
    prompt_image2=f"{style_prompt} {args.prompt2}",
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

os.makedirs(args.output_dir, exist_ok=True)
image1.save(os.path.join(args.output_dir, "generated_image1.png"))
image2.save(os.path.join(args.output_dir, "generated_image2.png"))

