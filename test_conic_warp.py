"""Test script for the conic mirror forward warp.

Inputs:
    input.png      — source rect-mirror image (e.g. the horse painting)
    output.png     — expected forward-warped disk image (paper reference)
    uv_conic.png   — expected UV map (paper reference, panel c)

Outputs (written next to this script):
    test_uv_actual.png      — our warp's UV plotted as paper's identity-UV
                              encoding: R = u_src, G = v_src, B = 0.
    test_warped_actual.png  — our forward conic warp applied to input.png.
    test_uv_expected.png    — the user-supplied uv_conic.png (resized).
    test_output_expected.png — the user-supplied output.png (resized).

We use the paper's ``identity UV'' encoding (R=u_src, G=v_src, B=0) because
that's exactly what the supplementary figure (a) and (b) show — the disk in
panel (c) is just the rect-mirror identity UV resampled through the polar
warp. So if our UV map matches uv_conic.png and our warped image matches
output.png, we know the warp itself is paper-correct.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'diffusers/src'))

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.stable_diffusion_3.lod_new import (
    create_conic_mirror_warp, view_simple, view_lod,
)


HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, size):
    img = Image.open(path).convert('RGB').resize(size, Image.LANCZOS)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def save(t, path):
    arr = (t.detach().squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f'  wrote {path}')


def uv_as_paper_rgb(warp_t, mask_t):
    """Encode the warp UV as the paper's identity UV: R = u_src, G = v_src,
    B = 0, with black outside the active mask."""
    uv = warp_t[0, :2].clamp(0, 1).cpu().numpy()
    rgb = np.stack([uv[0], uv[1], np.zeros_like(uv[0])], axis=-1)
    if mask_t is not None:
        m = mask_t[0, 0].clamp(0, 1).cpu().numpy()[..., None]
        rgb = rgb * m
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def main():
    H = W = 1024
    print(f'Building forward conic warp at {H}x{W}...')
    warp_f, mask_f = create_conic_mirror_warp(H, W, inverse=False)
    print(f'  warp:  {tuple(warp_f.shape)}    mask: {tuple(mask_f.shape)}')

    # 1. Plot the warp's UV with the paper's encoding.
    uv_rgb = uv_as_paper_rgb(warp_f, mask_f)
    save(uv_rgb, os.path.join(HERE, 'test_uv_actual.png'))

    # 2. Forward-warp the input image and save.
    src_path = os.path.join(HERE, 'input.png')
    if os.path.exists(src_path):
        src = load(src_path, (W, H))
        # forward conic in this codebase = source(rect) → dest(disk)
        warped_lod    = view_lod(src, warp_f, leveln=5, padding_mode='border')
        warped_simple = view_simple(src, warp_f)
        # mask off outside-disk so the comparison isn't dominated by corners.
        m = mask_f.expand(-1, 3, -1, -1)
        save((warped_lod    * m).clamp(0, 1), os.path.join(HERE, 'test_warped_actual_lod.png'))
        save((warped_simple * m).clamp(0, 1), os.path.join(HERE, 'test_warped_actual_simple.png'))

    # 3. Resave the user-supplied references at our resolution for easy A/B.
    for name in ('uv_conic.png', 'output.png'):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            ref = load(p, (W, H))
            tag = 'test_uv_expected.png' if name == 'uv_conic.png' else 'test_output_expected.png'
            save(ref, os.path.join(HERE, tag))

    # 4. Numeric comparisons (clipped to the disk mask so backgrounds don't
    #    dominate).
    if os.path.exists(os.path.join(HERE, 'uv_conic.png')):
        ref_uv = load(os.path.join(HERE, 'uv_conic.png'), (W, H))
        m = mask_f.expand(-1, 3, -1, -1)
        diff = ((uv_rgb - ref_uv).abs() * m).mean().item()
        print(f'\nUV map mean abs diff vs uv_conic.png (disk-masked): {diff:.4f}')

    if os.path.exists(os.path.join(HERE, 'output.png')) and os.path.exists(os.path.join(HERE, 'input.png')):
        ref_out = load(os.path.join(HERE, 'output.png'), (W, H))
        m = mask_f.expand(-1, 3, -1, -1)
        diff_lod = ((warped_lod * m - ref_out * m).abs()).mean().item()
        diff_simple = ((warped_simple * m - ref_out * m).abs()).mean().item()
        print(f'Warped image mean abs diff vs output.png (disk-masked):')
        print(f'  view_lod    : {diff_lod:.4f}')
        print(f'  view_simple : {diff_simple:.4f}')


if __name__ == '__main__':
    main()
