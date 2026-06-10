"""Sanity checks for the rewritten inner conic warp (annulus encoding).

The physical setup: an apex-up cone mirror stands on the print, covering
the central disk r < R_base. The visible reflection encodes view 2 in the
annulus [R_base, R_out] OUTSIDE the base circle (center <-> rim inversion,
azimuth preserved). These checks assert exactly that:

  1. The inverse warp (view 2 samples view 1) only reads source radii in
     [R_base, R_out]: view-2 center -> R_out, view-2 rim -> R_base.
  2. The forward warp leaves view 1 untouched inside the occluded disk
     and beyond the annulus, and its mask covers only the annulus.
  3. Round-trip: encode a test view 2 into the annulus (forward), read it
     back out (inverse) — content must survive inside the inscribed
     circle.

Writes test_annulus_*.png next to itself. Never touches the paper
reference images (input.png / output.png / uv_conic.png /
inverse_correct.png).
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'diffusers/src'))

import numpy as np
import torch
from PIL import Image

from diffusers.pipelines.stable_diffusion_3.lod_new import (
    create_conic_inner_mirror_warp, view_simple, view_lod,
    laplacian_warp_inverse,
)

HERE = os.path.dirname(os.path.abspath(__file__))
H = W = 1024
RADIUS_RATIO = 0.27
OUTER_RATIO = 0.475
R_BASE = RADIUS_RATIO * min(H, W)
R_OUT = OUTER_RATIO * min(H, W)
REACH = 0.5 * min(H, W)
# The warp's sampling region and mask ramp extend one feather width past
# both rims by design (so the rim blends have content on both sides).
FEATHER = max(1.0, 0.08 * (R_OUT - R_BASE))
MARGIN = FEATHER + 2.0


def save(t, name):
    arr = (t.detach().squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(HERE, name))
    print(f'  wrote {name}')


def src_radius(warp):
    """Source-sample radius (px) per dest pixel from a UV warp field."""
    u = warp[0, 0] * (W - 1)
    v = warp[0, 1] * (H - 1)
    return torch.sqrt((u - (W - 1) / 2.0) ** 2 + (v - (H - 1) / 2.0) ** 2)


def main():
    ok = True

    warp_f, mask_f = create_conic_inner_mirror_warp(
        H, W, radius_ratio=RADIUS_RATIO, outer_ratio=OUTER_RATIO, inverse=False)
    warp_i, mask_i = create_conic_inner_mirror_warp(
        H, W, radius_ratio=RADIUS_RATIO, outer_ratio=OUTER_RATIO, inverse=True)

    cy = (H - 1) / 2.0
    yy = torch.arange(H).float().view(-1, 1).expand(H, W)
    xx = torch.arange(W).float().view(1, -1).expand(H, W)
    r_dst = torch.sqrt((xx - cy) ** 2 + (yy - cy) ** 2)

    # --- 1. Inverse warp samples ONLY the annulus -----------------------
    r_src_i = src_radius(warp_i)
    inside_reach = r_dst <= REACH - 1.0
    lo = r_src_i[inside_reach].min().item()
    hi = r_src_i[inside_reach].max().item()
    print(f'inverse source radii over view-2 inscribed circle: '
          f'[{lo:.1f}, {hi:.1f}] px (annulus = [{R_BASE:.1f}, {R_OUT:.1f}])')
    if not (R_BASE - 2.0 <= lo and hi <= R_OUT + 2.0):
        print('  FAIL: inverse warp samples outside the annulus')
        ok = False
    # center -> far edge of annulus, rim -> cone base ring
    c = H // 2
    r_center = r_src_i[c - 1:c + 1, c - 1:c + 1].mean().item()
    rim_band = (r_dst > REACH - 3.0) & (r_dst < REACH - 1.0)
    r_rim = r_src_i[rim_band].mean().item()
    print(f'view-2 center samples r={r_center:.1f} (expect ~{R_OUT:.1f}); '
          f'view-2 rim samples r={r_rim:.1f} (expect ~{R_BASE:.1f})')
    if abs(r_center - R_OUT) > 3.0 or abs(r_rim - R_BASE) > 3.0:
        print('  FAIL: center<->rim inversion endpoints wrong')
        ok = False

    # --- 2. Forward warp: identity in the occluded disk + outside ------
    rng = np.random.RandomState(0)
    test1 = torch.from_numpy(rng.rand(1, 3, H, W).astype(np.float32))
    fwd = view_simple(test1, warp_f)
    occluded = r_dst < R_BASE - MARGIN
    beyond = r_dst > R_OUT + MARGIN
    d_occ = (fwd - test1)[..., occluded].abs().max().item()
    d_out = (fwd - test1)[..., beyond].abs().max().item()
    print(f'forward warp identity error: occluded disk {d_occ:.2e}, '
          f'beyond annulus {d_out:.2e}')
    if d_occ > 1e-4 or d_out > 1e-4:
        print('  FAIL: forward warp modifies pass-through regions')
        ok = False
    m = mask_f[0, 0]
    if m[occluded].max().item() > 0.0 or m[beyond].max().item() > 0.0:
        print('  FAIL: forward mask extends outside the annulus')
        ok = False
    annulus_core = (r_dst > R_BASE + 20) & (r_dst < R_OUT - 20)
    if m[annulus_core].min().item() < 1.0:
        print('  FAIL: forward mask not solid over the annulus core')
        ok = False
    print(f'forward mask: solid over annulus core, zero elsewhere — ok')

    # --- 3. Round-trip view2 -> annulus -> view2 ------------------------
    # Use a smooth radial/azimuthal test card as "view 2".
    th = torch.atan2(yy - cy, xx - cy)
    view2 = torch.stack([
        0.5 + 0.5 * torch.cos(3 * th),
        (r_dst / REACH).clamp(0, 1),
        0.5 + 0.5 * torch.sin(2 * th),
    ]).unsqueeze(0)
    encoded = laplacian_warp_inverse(view2, warp_i, leveln=6)
    # Confine to the annulus over a neutral print (as the pipeline does).
    m3 = mask_f.expand(-1, 3, -1, -1)
    print1 = torch.full_like(view2, 0.5)
    print1 = m3 * encoded + (1.0 - m3) * print1
    decoded = view_lod(print1, warp_i, leveln=6, padding_mode='border')
    core = (r_dst < REACH * 0.92) & (r_dst > 8.0)
    err = (decoded - view2)[..., core].abs().mean().item()
    print(f'round-trip (view2 -> annulus print -> view2) mean abs err '
          f'over inscribed circle: {err:.4f}')
    if err > 0.08:
        print('  FAIL: round-trip error too high')
        ok = False

    save(view2, 'test_annulus_view2.png')
    save(print1, 'test_annulus_print.png')
    save(decoded, 'test_annulus_decoded.png')

    print('\nPASS' if ok else '\nFAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
