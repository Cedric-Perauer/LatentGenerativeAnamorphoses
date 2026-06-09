import argparse
import os
import torch
from diffusers import StableDiffusion3Pipeline

#transform_type = "vertical"
possible_transform_types = ["vertical", "horizontal", "90flip", "90rot", "135rot", "180rot", "jigsaw", "conic", "conic_global"]

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
parser.add_argument(
    "--model",
    default="stabilityai/stable-diffusion-3.5-medium",
    help="HF repo id or local path of the SD3.5-medium weights (e.g. an ungated mirror).",
)
parser.add_argument(
    "--variant",
    default=None,
    help="Optional weights variant to load (e.g. 'fp16').",
)
parser.add_argument(
    "--conic-radius", type=float, default=0.27,
    help="[conic] circle radius as a fraction of min(H, W).",
)
parser.add_argument(
    "--conic-view2-weight", type=float, default=0.65,
    help="[conic] blend weight of view 2 inside the circle (0.5 = symmetric; "
         "higher makes the hidden image stronger but more visible in view 1).",
)
parser.add_argument(
    "--conic-view2-refine", type=float, default=0.8,
    help="[conic] SDEdit refinement strength for the final image2 "
         "(0 disables; higher = prettier but less faithful to the exact "
         "inverse warp).",
)
args = parser.parse_args()

style_prompt = args.style_prompt
transform_type = args.transform
seed = args.seed

pipe = StableDiffusion3Pipeline.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, variant=args.variant
).to("cuda")
pipe._conic_radius_ratio = args.conic_radius
pipe._conic_view2_weight = args.conic_view2_weight
pipe._conic_view2_refine = args.conic_view2_refine

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

if transform_type == "conic":
    # The conic image2 is derived by magnifying the inner circle of
    # image1, so its effective resolution is the circle diameter — save
    # it at that native size rather than blown up to the full canvas.
    from PIL import Image as _PILImage
    d = max(2, int(round(2 * args.conic_radius * 1024 / 2)) * 2)
    image2 = image2.resize((d, d), _PILImage.LANCZOS)
image2.save(os.path.join(args.output_dir, "generated_image2.png"))


def _save_warp_diagnostics(out_dir, transform, h=1024, w=1024):
    """Render and save the UV warp map, the blend mask, and the LOD map
    used by the Laplacian-pyramid warping path. Layout/coloring matches
    the LookingGlass paper figure: outside the active mask the UV map is
    rendered as black; the LOD map is shown with a viridis colormap."""
    from PIL import Image as _PILImage
    import numpy as _np
    import torch as _torch

    try:
        from diffusers.pipelines.stable_diffusion_3.lod_new import (
            create_identity_warp,
            create_vertical_flip_warp,
            create_circular_rotation_warp,
            create_conic_mirror_warp,
            create_conic_inner_mirror_warp,
            create_jigsaw_warp,
            _compute_lod_max_col,
        )
    except Exception as e:
        print(f"[warp viz] skipped: {e}")
        return

    def _uv_to_rgb(warp_t, mask_t):
        # Paper's literal encoding (matches resources/suppl_images-cone_view
        # panel b "identity UV" and panel c "mirror view UV"): R = u_src,
        # G = v_src, B = 0 — i.e. just plot the (u, v) source coordinates
        # directly. Black outside the active mask.
        uv = warp_t[0, :2].clamp(0, 1).cpu().numpy()
        rgb = _np.stack([uv[0], uv[1], _np.zeros_like(uv[0])], axis=-1)
        if mask_t is not None:
            m = mask_t[0, 0].clamp(0, 1).cpu().numpy()[..., None]
            rgb = rgb * m
        return (rgb * 255).clip(0, 255).astype(_np.uint8)

    def _mask_to_rgb(mask_t):
        if mask_t is None:
            return _np.full((h, w, 3), 200, dtype=_np.uint8)
        m = mask_t[0, 0].clamp(0, 1).cpu().numpy()
        return (_np.stack([m, m, m], axis=-1) * 255).astype(_np.uint8)

    def _lod_to_rgb(warp_t, mask_t, max_lod=4):
        # viridis-style colormap (5-stop linear interp) of the per-pixel LOD.
        mapping = float(max(h, w)) * warp_t[:, :2]
        lod = _compute_lod_max_col(mapping, maxLOD=max_lod).cpu().numpy()
        lod_n = (lod / max_lod).clip(0, 1)
        stops = _np.array([
            [0.267, 0.005, 0.329],   # 0 viridis dark purple
            [0.282, 0.140, 0.457],   # 0.25
            [0.190, 0.408, 0.556],   # 0.5
            [0.208, 0.718, 0.473],   # 0.75
            [0.993, 0.906, 0.143],   # 1.0 yellow
        ])
        # Linear interp between stops
        scaled = lod_n * 4.0
        i = _np.floor(scaled).clip(0, 3).astype(_np.int32)
        f = (scaled - i)[..., None]
        rgb = stops[i] * (1 - f) + stops[i + 1] * f
        if mask_t is not None:
            m = mask_t[0, 0].clamp(0, 1).cpu().numpy()[..., None]
            rgb = rgb * m + (1 - m) * stops[0]
        return (rgb * 255).clip(0, 255).astype(_np.uint8)

    if transform == "conic":
        warp_f, mask_f = create_conic_inner_mirror_warp(h, w, inverse=False)
        warp_i, mask_i = create_conic_inner_mirror_warp(h, w, inverse=True)
    elif transform == "conic_global":
        warp_f, mask_f = create_conic_mirror_warp(h, w, inverse=False)
        warp_i, mask_i = create_conic_mirror_warp(h, w, inverse=True)
    elif transform in ("90rot", "135rot", "180rot"):
        angle = float(transform.replace("rot", ""))
        warp_f, mask_f = create_circular_rotation_warp(h, w, angle, radius_ratio=0.45)
        warp_i, mask_i = create_circular_rotation_warp(h, w, -angle, radius_ratio=0.45)
    elif transform == "jigsaw":
        warp_f = create_jigsaw_warp(h, w, seed=42, inverse=False)
        warp_i = create_jigsaw_warp(h, w, seed=42, inverse=True)
        mask_f = mask_i = None
    elif transform == "vertical":
        warp_f = warp_i = create_vertical_flip_warp(h, w)
        mask_f = mask_i = None
    else:
        warp_f = warp_i = create_identity_warp(h, w)
        mask_f = mask_i = None

    _PILImage.fromarray(_uv_to_rgb(warp_f, mask_f)).save(os.path.join(out_dir, "warp_uv_forward.png"))
    _PILImage.fromarray(_uv_to_rgb(warp_i, mask_i)).save(os.path.join(out_dir, "warp_uv_inverse.png"))
    _PILImage.fromarray(_mask_to_rgb(mask_f)).save(os.path.join(out_dir, "warp_mask_forward.png"))
    _PILImage.fromarray(_mask_to_rgb(mask_i)).save(os.path.join(out_dir, "warp_mask_inverse.png"))
    _PILImage.fromarray(_lod_to_rgb(warp_f, mask_f)).save(os.path.join(out_dir, "warp_lod_forward.png"))
    _PILImage.fromarray(_lod_to_rgb(warp_i, mask_i)).save(os.path.join(out_dir, "warp_lod_inverse.png"))
    print(f"[warp viz] wrote UV + mask + LOD to {out_dir}")


_save_warp_diagnostics(args.output_dir, transform_type)

