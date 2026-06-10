import argparse
import os
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

possible_transform_types = ["vertical", "horizontal", "90flip", "90rot", "135rot", "180rot", "jigsaw"]

parser = argparse.ArgumentParser(description="Generate Flux2 Dev dual-prompt anamorphic images.")
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
parser.add_argument("--num-inference-steps", type=int, default=28, help="Number of denoising steps.")
parser.add_argument("--guidance-scale", type=float, default=3.5, help="Embedded guidance scale for Flux.")
parser.add_argument("--true-cfg-scale", type=float, default=3.0,
                    help="True CFG scale (>1 enables real cond/uncond CFG like SD3.5; "
                         "set to 1.0 for the original embedded-guidance-only behavior).")
parser.add_argument("--negative-prompt", default="", help="Negative prompt for true CFG.")
parser.add_argument("--negative-prompt2", default=None,
                    help="Per-view negative prompt for the second view (defaults to --negative-prompt).")
parser.add_argument("--height", type=int, default=1024, help="Image height.")
parser.add_argument("--width", type=int, default=1024, help="Image width.")
parser.add_argument("--time-travel", type=int, default=2, help="Number of time-travel backward steps.")
parser.add_argument("--time-travel-range", type=int, nargs=2, default=[20, 80],
                    help="Percentage range for time-travel (start end).")
parser.add_argument("--denoise-last-steps", type=int, default=3,
                    help="Number of final steps for single-view denoising.")
parser.add_argument("--no-lwp", action="store_true", help="Disable Laplacian Weighted Pooling.")
parser.add_argument("--lwp-alpha", type=float, default=0.5,
                    help="Blend weight toward value-weighted average (0=plain average, 1=full vavg).")
parser.add_argument("--no-denoise-last", action="store_true", help="Disable final single-view denoising.")
args = parser.parse_args()

style_prompt = args.style_prompt
transform_type = args.transform
seed = args.seed

image1, image2 = pipe(
    prompt=f"{style_prompt} {args.prompt1}",
    prompt_image2=f"{style_prompt} {args.prompt2}",
    num_inference_steps=args.num_inference_steps,
    height=args.height,
    width=args.width,
    guidance_scale=args.guidance_scale,
    true_cfg_scale=args.true_cfg_scale,
    negative_prompt=args.negative_prompt,
    negative_prompt_image2=args.negative_prompt2,
    transform_type=transform_type,
    time_travel=args.time_travel,
    generator=torch.Generator("cuda").manual_seed(seed),
    time_travel_range=args.time_travel_range,
    denoise_last=not args.no_denoise_last,
    denoise_last_steps=args.denoise_last_steps,
    vis_intermediate=False,
    lwp=not args.no_lwp,
    lwp_alpha=args.lwp_alpha,
)

os.makedirs(args.output_dir, exist_ok=True)
image1.save(os.path.join(args.output_dir, "generated_image1.png"))
image2.save(os.path.join(args.output_dir, "generated_image2.png"))
