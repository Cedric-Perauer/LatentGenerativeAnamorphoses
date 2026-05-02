"""Test script for the conic mirror forward + inverse warps.

Inputs (DO NOT OVERWRITE):
    input.png            — source rect-mirror image (the horse painting).
    output.png           — expected forward-warped disk image (paper ref).
    uv_conic.png         — expected UV map (paper ref, panel c).
    inverse_correct.png  — benchmark for the inverse roundtrip: how the
                           paper's output.png should look after running
                           through the inverse conic warp. We compare
                           OUR inverse-warp result against this; we never
                           overwrite this file.

Outputs (written next to this script):
    test_uv_actual.png            — our warp's UV in the paper's encoding:
                                    R = u_src, G = v_src, B = 0.
    test_warped_actual_*.png      — our forward conic warp on input.png.
    test_uv_expected.png          — uv_conic.png at our resolution.
    test_output_expected.png      — output.png at our resolution.
    test_inverse_actual_*.png     — our inverse conic warp applied to the
                                    paper's output.png. Compared to
                                    inverse_correct.png as benchmark.
    test_roundtrip_self_*.png     — input → our forward → our inverse,
                                    a self-consistency check on our pair.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'diffusers/src'))

import numpy as np
import torch
import scipy.ndimage
from PIL import Image

from diffusers.pipelines.stable_diffusion_3.lod_new import (
    create_conic_mirror_warp, view_simple, view_lod,
    inverse_view, Laplacian2Gaussian,
)


def soften_inverse(recovered_t, mask_t, apex_sigma=60.0,
                   outside_blur_sigma=20.0, outside_fade_dist=40.0,
                   edge_blend_sigma=6.0):
    """Apply blur post-processing to the inverse-warp result, matching the
    paper's inverse_correct.png benchmark style.

    Three regions are handled:

    - **Inside** the disk silhouette (sharp horse content): uses the
      recovered image directly.
    - **Apex hole** at the rect-mirror centre (the only pixels that the
      inverse warp had to fill via scatter-then-interpolate, since the
      cone apex has no scatter contributions): blended in toward a
      heavy-blur version of the recovered content, weighted by distance
      from the mask boundary so the deepest interior gets the most blur
      and the surface stays sharp.
    - **Outside** the disk silhouette: a heavily blurred copy of the
      recovered content, alpha-faded to black over a distance
      ``outside_fade_dist``. This avoids the previous hard cutoff that
      revealed the inverse warp's edge streaks, and matches the soft
      "halo" the paper benchmark shows around its disk.

    All three regions are blended with a smooth mask transition so the
    boundary itself isn't visible as a discrete edge.

    Args:
        recovered_t: (1, 3, H, W) torch tensor — output of view_*.
        mask_t:      (1, 1, H, W) torch tensor — binary inverse mask
                     (1 inside the disk silhouette + apex hole, 0 outside).
        apex_sigma:  Gaussian sigma (px) for the apex blur.
        outside_blur_sigma: Gaussian sigma (px) for the outside blur.
        outside_fade_dist: distance (px) over which outside content fades
                           to black.
        edge_blend_sigma: Gaussian sigma (px) for the mask transition
                           between inside content and outside content.
    """
    img = recovered_t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype(np.float32)
    m   = mask_t[0, 0].clamp(0, 1).cpu().numpy().astype(np.float32)
    H, W = m.shape

    # Heavy-blur passes for apex hole and outside.
    img_apex_blur = np.stack([
        scipy.ndimage.gaussian_filter(img[..., c], sigma=apex_sigma)
        for c in range(3)
    ], axis=-1)
    img_outside_blur = np.stack([
        scipy.ndimage.gaussian_filter(img[..., c], sigma=outside_blur_sigma)
        for c in range(3)
    ], axis=-1)

    # The mask encodes three regions:
    #   m == 1.0  → native scatter (sharp content)
    #   m ≈ 0.5   → apex hole (no scatter, filled by interpolation → blur)
    #   m == 0.0  → outside the disk silhouette (handled separately below)
    # Use the apex-hole region directly as the blur weight rather than a
    # distance-transform heuristic — this places blur exactly where the
    # warp couldn't fill from scatter (the inner cone-apex circle in the
    # rect centre, plus any thin self-occlusion ring at the outer edge).
    apex_hole = ((m > 0.3) & (m < 0.7)).astype(np.float32)
    # Soften the binary apex-hole region into a continuous weight, with a
    # smaller sigma so the blur stays localized to the actual hole region.
    apex_w = scipy.ndimage.gaussian_filter(apex_hole, sigma=20.0)
    apex_w = np.clip(apex_w, 0.0, 1.0)
    inside_content = img * (1.0 - apex_w[..., None]) + img_apex_blur * apex_w[..., None]

    # Outside content: heavily blurred image, faded to black with distance
    # from the mask boundary (so far corners → black, near-boundary → blur).
    dist_outside = scipy.ndimage.distance_transform_edt(1 - (m > 0.5))
    outside_alpha = np.clip(1.0 - dist_outside / outside_fade_dist, 0.0, 1.0)
    # Smooth out the alpha so the fade itself doesn't have steps.
    outside_alpha = scipy.ndimage.gaussian_filter(outside_alpha, sigma=8.0)
    outside_alpha = np.clip(outside_alpha, 0.0, 1.0)
    outside_content = img_outside_blur * outside_alpha[..., None]

    # Smooth blend between inside and outside content so the boundary is
    # soft, not a discrete edge.
    soft_m = scipy.ndimage.gaussian_filter(m, sigma=edge_blend_sigma)
    soft_m = np.clip(soft_m, 0.0, 1.0)

    out = soft_m[..., None] * inside_content + (1.0 - soft_m[..., None]) * outside_content
    out = np.clip(out, 0, 1).astype(np.float32)
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)


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

    # 5. Inverse roundtrip: take the paper's expected disk output, run our
    #    custom inverse conic warp (scatter + multi-scale Gaussian
    #    push-pull fill), then post-process with `soften_inverse`:
    #      - apex region (deepest interior of the mask) gets a strong
    #        Gaussian blur, blended with sharp content based on depth;
    #      - outside the disk silhouette, the recovered content is
    #        heavily blurred and alpha-faded to black over a short
    #        distance (outside_fade_dist).
    #    This was visually closer to the benchmark than the pyramid-
    #    based inverse_view path for our specific eye/cone parameters.
    out_path  = os.path.join(HERE, 'output.png')
    in_path   = os.path.join(HERE, 'input.png')
    bench_path = os.path.join(HERE, 'inverse_correct.png')  # READ-ONLY benchmark
    if os.path.exists(out_path) and os.path.exists(in_path):
        print('\nBuilding inverse conic warp...')
        warp_i, mask_i = create_conic_mirror_warp(H, W, inverse=True)

        ref_disk  = load(out_path, (W, H))   # paper's canonical disk image
        ref_input = load(in_path,  (W, H))   # original rect mirror

        # Apply the inverse warp (disk → rect) to the paper's disk.
        recovered_lod    = view_lod(ref_disk, warp_i, leveln=5,
                                    padding_mode='border')
        recovered_simple = view_simple(ref_disk, warp_i)

        # Post-process: feathered alpha + apex blur + outside blur fade.
        recovered_lod_masked    = soften_inverse(recovered_lod,    mask_i)
        recovered_simple_masked = soften_inverse(recovered_simple, mask_i)

        save(recovered_lod_masked,    os.path.join(HERE, 'test_inverse_actual_lod.png'))
        save(recovered_simple_masked, os.path.join(HERE, 'test_inverse_actual_simple.png'))

        # Compare to the user-supplied benchmark, NOT to input.png — the
        # benchmark is what the inverse SHOULD look like (with masking +
        # blur fill for the apex hole). Don't write to inverse_correct.png.
        if os.path.exists(bench_path):
            bench = load(bench_path, (W, H))
            diff_lod_b    = (recovered_lod_masked    - bench).abs().mean().item()
            diff_simple_b = (recovered_simple_masked - bench).abs().mean().item()
            print(f'Inverse warp — mean abs diff vs inverse_correct.png benchmark:')
            print(f'  view_lod    : {diff_lod_b:.4f}')
            print(f'  view_simple : {diff_simple_b:.4f}')

        # And the raw recovery vs input.png (just for context).
        m_inv_3 = mask_i.expand(-1, 3, -1, -1)
        diff_lod_inv    = (recovered_lod_masked    - ref_input * m_inv_3).abs().mean().item()
        diff_simple_inv = (recovered_simple_masked - ref_input * m_inv_3).abs().mean().item()
        print(f'Inverse warp — mean abs diff vs input.png (mask-clipped):')
        print(f'  view_lod    : {diff_lod_inv:.4f}')
        print(f'  view_simple : {diff_simple_inv:.4f}')

        # Self round-trip through our forward + inverse warps (consistency).
        print('\nSelf round-trip through our warps (forward then inverse):')
        rt_simple = view_simple(view_simple(ref_input, warp_f), warp_i)
        rt_lod    = view_lod(view_lod(ref_input, warp_f, leveln=5,
                                      padding_mode='border'),
                             warp_i, leveln=5, padding_mode='border')
        rt_simple_m = soften_inverse(rt_simple, mask_i)
        rt_lod_m    = soften_inverse(rt_lod,    mask_i)
        save(rt_simple_m, os.path.join(HERE, 'test_roundtrip_self_simple.png'))
        save(rt_lod_m,    os.path.join(HERE, 'test_roundtrip_self_lod.png'))
        print(f'  view_simple round-trip mean abs diff: '
              f'{(rt_simple_m - ref_input * m_inv_3).abs().mean().item():.4f}')
        print(f'  view_lod    round-trip mean abs diff: '
              f'{(rt_lod_m    - ref_input * m_inv_3).abs().mean().item():.4f}')


if __name__ == '__main__':
    main()
