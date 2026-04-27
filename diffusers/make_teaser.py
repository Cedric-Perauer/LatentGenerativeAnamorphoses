"""
Teaser video for the LatentGenerativeAnamorphoses illusions.

Two modes are supported:

1. `--mode jigsaw` (default): generated_image1 dissolves into 16 puzzle pieces,
   each piece physically travels along a spline to the slot it occupies in
   generated_image2 (no content cross-fade — pieces keep their image1 colors
   the entire trip; the rearrangement itself is what produces image2).

2. `--mode inner_circle`: an inner disk of generated_image1 rotates by
   `--rotation` degrees (positive = counter-clockwise / "to the left") while
   the surround cross-fades to generated_image2, landing on image2 by t=1.
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from einops import einsum, rearrange
from scipy.optimize import linear_sum_assignment

# ---- paths (resolved relative to this file so the script is repo-portable) ----
ROOT = Path(__file__).resolve().parent
PUZZLE_DIR = ROOT / 'puzzle_4x4'
DEFAULT_IMG_DIR = ROOT / 'outputs' / 'plants_marily'

# ---- canvas / timing ----
IM_SIZE = 1024
CANVAS_SIZE = int(IM_SIZE * 1.5)
OUT_SIZE = 1024
img_offset = (CANVAS_SIZE - IM_SIZE) // 2
FPS = 30
HOLD1_S = 0.7
HOLD2_S = 1.2
ANIM_S = 4.6
N_FRAMES = int(round(FPS * (HOLD1_S + ANIM_S + HOLD2_S)))
HOLD1_FRAMES = int(round(FPS * HOLD1_S))
HOLD2_FRAMES = int(round(FPS * HOLD2_S))
ANIM_FRAMES = N_FRAMES - HOLD1_FRAMES - HOLD2_FRAMES

# ---- jigsaw-mode params ----
SCATTER_SEED = 0
SCATTER_PUSH = 240
SCATTER_JITTER = 35
SCATTER_END = 0.38
HOLD_END = 0.55


# ---- CLI ----
parser = argparse.ArgumentParser()
parser.add_argument('--img1', type=Path,
                    default=DEFAULT_IMG_DIR / 'generated_image1.png',
                    help='Path to source image 1 (the "before" view)')
parser.add_argument('--img2', type=Path,
                    default=DEFAULT_IMG_DIR / 'generated_image2.png',
                    help='Path to source image 2 (the "after" view)')
parser.add_argument('--out', type=Path, default=None,
                    help='Output mp4 path (default: <img1.parent>/teaser.mp4)')
parser.add_argument('--caption1', type=str, default='',
                    help='Optional caption shown beneath image1 during the opening hold')
parser.add_argument('--caption2', type=str, default='',
                    help='Optional caption shown beneath the final frame during the closing hold')
parser.add_argument('--mode', choices=['jigsaw', 'inner_circle', 'rotate'], default='jigsaw',
                    help='Which illusion type to animate')
parser.add_argument('--rotation', type=float, default=90.0,
                    help='[inner_circle / rotate] degrees of rotation to animate; +90 = CCW ("to the left"), -90 = CW')
parser.add_argument('--radius', type=int, default=None,
                    help='[inner_circle] disk radius in px; auto-detected from img1↔img2 diff envelope when omitted')
parser.add_argument('--feather', type=int, default=4,
                    help='[inner_circle] disk-edge feather radius in pixels for soft compositing')
args = parser.parse_args()

OUT_PATH = args.out if args.out is not None else args.img1.parent / 'teaser.mp4'
print(f'Mode: {args.mode}\nSource 1: {args.img1}\nSource 2: {args.img2}\nOutput: {OUT_PATH}')


# ---- load images ----
img1 = np.array(Image.open(args.img1).convert('RGB'))
img2 = np.array(Image.open(args.img2).convert('RGB'))
assert img1.shape == img2.shape == (IM_SIZE, IM_SIZE, 3), \
    f'Both images must be {IM_SIZE}x{IM_SIZE} RGB; got {img1.shape} and {img2.shape}'


# ---- caption rendering (shared) ----
CAPTION_FONT_SIZE = 60
CAPTION_COLOR = (40, 40, 40)
CAPTION_FONT_CANDIDATES = [
    '/System/Library/Fonts/Helvetica.ttc',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/Library/Fonts/Arial.ttf',
]


def _load_caption_font(size):
    for p in CAPTION_FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


CAPTION_FONT = _load_caption_font(CAPTION_FONT_SIZE)
CAPTION_CENTER_Y = (img_offset + IM_SIZE + CANVAS_SIZE) // 2


def draw_caption(rgb_canvas, text):
    if not text:
        return rgb_canvas
    draw = ImageDraw.Draw(rgb_canvas)
    bbox = draw.textbbox((0, 0), text, font=CAPTION_FONT)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (CANVAS_SIZE - w) // 2
    y = CAPTION_CENTER_Y - bbox[1] - h // 2
    draw.text((x, y), text, font=CAPTION_FONT, fill=CAPTION_COLOR)
    return rgb_canvas


def smoothstep(x):
    return x * x * (3 - 2 * x)


def static_frame(img, caption=''):
    canvas = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    canvas.paste(Image.fromarray(img), (img_offset, img_offset))
    draw_caption(canvas, caption)
    if OUT_SIZE != CANVAS_SIZE:
        canvas = canvas.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    return np.array(canvas)


# =====================================================================
# Mode-specific setup + render_frame_at_t implementation
# =====================================================================

if args.mode == 'jigsaw':
    # ---- load piece masks ----
    def load_piece_masks():
        masks = []
        for name in ('4x4_corner', '4x4_inner', '4x4_edge1', '4x4_edge2'):
            m = np.array(Image.open(PUZZLE_DIR / f'{name}_{IM_SIZE}.png'))[..., 0] // 255
            for k in range(4):
                masks.append(np.rot90(m, k=-k))
        return np.stack(masks)

    masks = load_piece_masks()

    def extract_pieces(img):
        pieces = []
        for m in masks:
            rgba = np.concatenate([img, (m[..., None] * 255).astype(np.uint8)], axis=2)
            ys, xs = np.where(m > 0)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            pieces.append(Image.fromarray(rgba[y0:y1, x0:x1]))
        return pieces

    pieces1 = extract_pieces(img1)
    pieces2 = extract_pieces(img2)

    base1 = [p.rotate(90 * (i % 4), resample=Image.BILINEAR, expand=True)
             for i, p in enumerate(pieces1)]
    base2 = [p.rotate(90 * (i % 4), resample=Image.BILINEAR, expand=True)
             for i, p in enumerate(pieces2)]

    base1_arr = [np.asarray(p, dtype=np.float32) for p in base1]
    base2_arr = [np.asarray(p, dtype=np.float32) for p in base2]

    INCOMPATIBLE_COST = 1e6

    def piece_distance(a, b):
        if a.shape != b.shape:
            return INCOMPATIBLE_COST
        common = (a[..., 3] > 0) & (b[..., 3] > 0)
        if not common.any():
            return INCOMPATIBLE_COST
        return float(np.abs(a[common, :3] - b[common, :3]).mean())

    cost_matrix = np.array([[piece_distance(base1_arr[i], base2_arr[j])
                             for j in range(16)] for i in range(16)])
    rows, cols = linear_sum_assignment(cost_matrix)
    perm = np.empty(16, dtype=int)
    perm[rows] = cols
    total_cost = cost_matrix[rows, cols].sum()
    print(f'Recovered perm: {perm.tolist()}  total RGB dist: {total_cost:.2f}')

    # ---- start / end positions ----
    corner_start = np.array([-1.5, -1.5])
    inner_start = np.array([-0.5, -0.5])
    edge_e_start = np.array([-1.5, -0.5])
    edge_f_start = np.array([-1.5, 0.5])
    base_starts = np.stack([corner_start, inner_start, edge_e_start, edge_f_start])

    rot_mats = []
    for theta in -np.arange(4) * np.pi / 2:
        rot_mats.append(np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta),  np.cos(theta)]]))
    rot_mats = np.stack(rot_mats)

    start_locs = einsum(base_starts, rot_mats, 'start i, rot j i -> start rot j')
    start_locs = rearrange(start_locs, 'start rot j -> (start rot) j')

    thetas = np.tile(np.arange(4) * -90, 4)[:, None]
    start_locs = np.concatenate([start_locs, thetas], axis=1).astype(np.float64)
    end_locs = start_locs[perm].copy()

    start_locs[:, :2] = (start_locs[:, :2] + 2) * (IM_SIZE / 4) + img_offset
    end_locs[:, :2] = (end_locs[:, :2] + 2) * (IM_SIZE / 4) + img_offset

    theta_delta = end_locs[:, 2] - start_locs[:, 2]
    theta_delta = ((theta_delta + 180) % 360) - 180

    canvas_center = np.array([CANVAS_SIZE / 2, CANVAS_SIZE / 2])
    delta_from_center = start_locs[:, :2] - canvas_center
    radii = np.linalg.norm(delta_from_center, axis=1, keepdims=True)
    out_dirs = delta_from_center / np.maximum(radii, 1e-6)
    np.random.seed(SCATTER_SEED)
    jitter = np.random.randn(16, 2) * SCATTER_JITTER
    scatter_locs = canvas_center + out_dirs * (radii + SCATTER_PUSH) + jitter

    PIECE_HALF = IM_SIZE // 4 // 2  # 128

    def piece_at_t(i, t):
        start_yx = start_locs[i, :2]
        end_yx = end_locs[i, :2]
        scat_yx = scatter_locs[i]
        if t <= SCATTER_END:
            s = smoothstep(t / SCATTER_END)
            pos = (1 - s) * start_yx + s * scat_yx
        elif t <= HOLD_END:
            pos = scat_yx
        else:
            s = smoothstep((t - HOLD_END) / max(1e-6, 1.0 - HOLD_END))
            pos = (1 - s) * scat_yx + s * end_yx
        theta = float(start_locs[i, 2] + smoothstep(t) * theta_delta[i])
        return int(round(pos[1])), int(round(pos[0])), theta

    def paste_piece(canvas, piece, x, y, theta):
        layer = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 0))
        layer.paste(piece, (x - PIECE_HALF, y - PIECE_HALF), piece)
        if theta != 0.0:
            layer = layer.rotate(theta, resample=Image.BILINEAR, center=(x, y))
        canvas.alpha_composite(layer)

    def render_frame_at_t(t, caption=''):
        canvas = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))
        for i in range(16):
            x, y, theta = piece_at_t(i, t)
            paste_piece(canvas, base1[i], x, y, theta)
        canvas_rgb = canvas.convert('RGB')
        draw_caption(canvas_rgb, caption)
        if OUT_SIZE != CANVAS_SIZE:
            canvas_rgb = canvas_rgb.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
        return np.array(canvas_rgb)

elif args.mode == 'inner_circle':
    # Auto-detect the disk radius from where img1 and img2 actually differ —
    # outside the manipulated disk the two images are bytewise identical.
    if args.radius is None:
        diff_max = np.abs(img1.astype(np.int16) - img2.astype(np.int16)).max(axis=-1)
        cy = (IM_SIZE - 1) / 2.0
        yy0, xx0 = np.indices((IM_SIZE, IM_SIZE), dtype=np.float32)
        dist0 = np.sqrt((yy0 - cy) ** 2 + (xx0 - cy) ** 2)
        if (diff_max > 0).any():
            args.radius = int(np.ceil(dist0[diff_max > 0].max())) + 2
        else:
            args.radius = IM_SIZE // 2
        print(f'Auto-detected inner-disk radius: {args.radius}px')

    # Feathered alpha mask: 1 inside the disk, 0 outside, smooth at the boundary
    cy = (IM_SIZE - 1) / 2.0
    cx = (IM_SIZE - 1) / 2.0
    yy, xx = np.indices((IM_SIZE, IM_SIZE), dtype=np.float32)
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disk_alpha = np.clip((args.radius - dist) / max(1, args.feather) + 0.5, 0.0, 1.0)
    disk_alpha = disk_alpha.astype(np.float32)[..., None]  # (H, W, 1)
    img1_pil = Image.fromarray(img1)
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    print(f'Inner-circle mode: rotating disk r={args.radius}px by {args.rotation:+.1f}° '
          f'(positive = CCW), feather={args.feather}px')

    def render_frame_at_t(t, caption=''):
        angle = t * args.rotation
        # PIL rotates CCW for positive angles, around the image center by default
        rotated = np.array(img1_pil.rotate(angle, resample=Image.BICUBIC),
                           dtype=np.float32)
        # Inner and outer both cross-fade to image2 so we land exactly on image2
        # at t=1 regardless of how many degrees the disk is animated through.
        inner = (1.0 - t) * rotated + t * img2_f
        outer = (1.0 - t) * img1_f + t * img2_f
        composed = inner * disk_alpha + outer * (1.0 - disk_alpha)
        composed = np.clip(composed, 0.0, 255.0).astype(np.uint8)

        canvas = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
        canvas.paste(Image.fromarray(composed), (img_offset, img_offset))
        draw_caption(canvas, caption)
        if OUT_SIZE != CANVAS_SIZE:
            canvas = canvas.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
        return np.array(canvas)

elif args.mode == 'rotate':
    # Whole-image rotation. Two layers — image1 rotated to current angle, image2
    # rotated by (current angle - target rotation) — cross-faded so we anchor
    # exactly on image1 at t=0 and image2 at t=1. For pixel-exact 90° pairs, the
    # two layers agree at every intermediate angle so the cross-fade is invisible.
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)

    def _embed(img_pil):
        canvas = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
        canvas.paste(img_pil, (img_offset, img_offset))
        return canvas

    canvas1_template = _embed(img1_pil)
    canvas2_template = _embed(img2_pil)
    print(f'Rotate mode: full-image rotation of {args.rotation:+.1f}° '
          f'(positive = CCW)')

    def render_frame_at_t(t, caption=''):
        angle = t * args.rotation
        rot1 = np.array(canvas1_template.rotate(angle, resample=Image.BICUBIC,
                                                fillcolor=(255, 255, 255)),
                        dtype=np.float32)
        rot2 = np.array(canvas2_template.rotate(angle - args.rotation,
                                                resample=Image.BICUBIC,
                                                fillcolor=(255, 255, 255)),
                        dtype=np.float32)
        composed = (1.0 - t) * rot1 + t * rot2
        composed = np.clip(composed, 0.0, 255.0).astype(np.uint8)

        canvas = Image.fromarray(composed)
        draw_caption(canvas, caption)
        if OUT_SIZE != CANVAS_SIZE:
            canvas = canvas.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
        return np.array(canvas)


def main():
    print(f'Rendering {N_FRAMES} frames '
          f'({HOLD1_FRAMES} hold1 + {ANIM_FRAMES} anim + {HOLD2_FRAMES} hold2) '
          f'at {FPS}fps -> {OUT_PATH}')
    frame1_static = static_frame(img1, caption=args.caption1)
    final_frame = render_frame_at_t(1.0, caption=args.caption2)

    frames = np.empty((N_FRAMES, OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)
    for f in range(N_FRAMES):
        if f < HOLD1_FRAMES:
            frames[f] = frame1_static
        elif f >= HOLD1_FRAMES + ANIM_FRAMES:
            frames[f] = final_frame
        else:
            anim_t = (f - HOLD1_FRAMES) / max(1, ANIM_FRAMES - 1)
            frames[f] = render_frame_at_t(smoothstep(anim_t))
        if (f + 1) % 10 == 0 or f == N_FRAMES - 1:
            print(f'  frame {f + 1}/{N_FRAMES}')

    iio.imwrite(
        OUT_PATH,
        frames,
        fps=FPS,
        codec='libx264',
        pixelformat='yuv420p',
        macro_block_size=1,
        quality=8,
    )
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
