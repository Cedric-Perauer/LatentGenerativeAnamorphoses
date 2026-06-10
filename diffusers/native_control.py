"""Native FLUX.1-dev sampling control (no anamorphosis loop) using the standard
Euler flow-matching steps, to disentangle model style from pipeline degradation."""
import argparse
import os
import numpy as np
import torch
from diffusers import FluxPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num-inference-steps", type=int, default=28)
parser.add_argument("--guidance-scale", type=float, default=3.5)
parser.add_argument("--output", required=True)
args = parser.parse_args()

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

device = "cuda"
height = width = 1024
generator = torch.Generator(device).manual_seed(args.seed)

prompt_embeds, pooled, text_ids = pipe.encode_prompt(
    prompt=args.prompt, device=device, max_sequence_length=512
)

num_channels_latents = pipe.transformer.config.in_channels // 4
latents, latent_image_ids = pipe.prepare_latents(
    1, num_channels_latents, height, width, prompt_embeds.dtype, device, generator
)

from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

sigmas = np.linspace(1.0, 1 / args.num_inference_steps, args.num_inference_steps)
mu = calculate_shift(
    latents.shape[1],
    pipe.scheduler.config.get("base_image_seq_len", 256),
    pipe.scheduler.config.get("max_image_seq_len", 4096),
    pipe.scheduler.config.get("base_shift", 0.5),
    pipe.scheduler.config.get("max_shift", 1.15),
)
timesteps, _ = retrieve_timesteps(pipe.scheduler, args.num_inference_steps, device, sigmas=sigmas, mu=mu)

guidance = torch.full([1], args.guidance_scale, device=device, dtype=torch.float32)

torch.set_grad_enabled(False)

for i, t in enumerate(timesteps):
    timestep = t.expand(latents.shape[0]).to(latents.dtype)
    noise_pred = pipe.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=guidance,
        pooled_projections=pooled,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]
    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
image = pipe.vae.decode(latents, return_dict=False)[0]
image = pipe.image_processor.postprocess(image, output_type="pil")[0]
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
image.save(args.output)
print("saved", args.output)
