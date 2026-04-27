"""
Teaser video for the plants_marily jigsaw illusion.

Renders generated_image1 dissolving into 16 puzzle pieces, each piece
physically traveling along a spline to the slot it occupies in
generated_image2 (no content cross-fade — pieces keep their image1 colors
the entire trip; the rearrangement itself is what produces image2).
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

# Defaults (overridable via CLI). Pair `(img1, img2, out)` resolved at runtime.
DEFAULT_IMG_DIR = ROOT / 'outputs' / 'plants_marily'

# ---- params ----
IM_SIZE = 1024
CANVAS_SIZE = int(IM_SIZE * 1.5)
OUT_SIZE = 1024
FPS = 30
HOLD1_S = 0.7
HOLD2_S = 1.2
ANIM_S = 4.6
N_FRAMES = int(round(FPS * (HOLD1_S + ANIM_S + HOLD2_S)))
HOLD1_FRAMES = int(round(FPS * HOLD1_S))
HOLD2_FRAMES = int(round(FPS * HOLD2_S))
ANIM_FRAMES = N_FRAMES - HOLD1_FRAMES - HOLD2_FRAMES

SCATTER_SEED = 0
SCATTER_PUSH = 240         # pixels each piece is pushed radially outward at the scatter point
SCATTER_JITTER = 35        # per-piece jitter so the scatter ring isn't perfectly tidy
SCATTER_END = 0.38         # fraction of anim where pieces finish flying outward
HOLD_END = 0.55            # fraction where pieces start flying back in to image2 slots


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
                    help='Optional caption shown beneath the assembled puzzle during the closing hold')
args = parser.parse_args()

OUT_PATH = args.out if args.out is not None else args.img1.parent / 'teaser.mp4'
print(f'Source 1: {args.img1}\nSource 2: {args.img2}\nOutput: {OUT_PATH}')

# ---- load images ----
img1 = np.array(Image.open(args.img1).convert('RGB'))
img2 = np.array(Image.open(args.img2).convert('RGB'))
assert img1.shape == img2.shape == (IM_SIZE, IM_SIZE, 3), \
    f'Both images must be {IM_SIZE}x{IM_SIZE} RGB; got {img1.shape} and {img2.shape}'


# ---- load piece masks ----
def load_piece_masks():
    masks = []
    for name in ('4x4_corner', '4x4_inner', '4x4_edge1', '4x4_edge2'):
        m = np.array(Image.open(PUZZLE_DIR / f'{name}_{IM_SIZE}.png'))[..., 0] // 255
        for k in range(4):
            masks.append(np.rot90(m, k=-k))
    return np.stack(masks)


masks = load_piece_masks()


# ---- extract pieces and rotate to base orientation ----
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


# ---- recover the permutation that takes image1's pieces to image2's slots ----
# We compare base-rotated pieces by mean abs RGB distance over the alpha-covered
# region, then run Hungarian (linear-sum-assignment) across all 16x16 pairs.
# Shape-incompatible pairs (e.g. corner vs edge) get a large penalty so they
# never get selected; this still allows edge1<->edge2 swaps because those two
# piece types share the same base shape (per the project's permutation docs).
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


# ---- start / end positions on the canvas (mirrors view_jigsaw.make_frame) ----
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
end_locs = start_locs[perm].copy()  # piece i ends at slot perm[i]

img_offset = (CANVAS_SIZE - IM_SIZE) // 2
start_locs[:, :2] = (start_locs[:, :2] + 2) * (IM_SIZE / 4) + img_offset
end_locs[:, :2] = (end_locs[:, :2] + 2) * (IM_SIZE / 4) + img_offset

# Take the shortest angular path for each piece
theta_delta = end_locs[:, 2] - start_locs[:, 2]
theta_delta = ((theta_delta + 180) % 360) - 180  # wrap to [-180, 180]


# ---- scatter positions: each piece pushed radially outward from canvas center ----
canvas_center = np.array([CANVAS_SIZE / 2, CANVAS_SIZE / 2])
delta_from_center = start_locs[:, :2] - canvas_center
radii = np.linalg.norm(delta_from_center, axis=1, keepdims=True)
out_dirs = delta_from_center / np.maximum(radii, 1e-6)
np.random.seed(SCATTER_SEED)
jitter = np.random.randn(16, 2) * SCATTER_JITTER
scatter_locs = canvas_center + out_dirs * (radii + SCATTER_PUSH) + jitter

PIECE_HALF = IM_SIZE // 4 // 2  # 128


# ---- caption rendering ----
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
# Center the caption vertically in the bottom margin (between image bottom and canvas bottom)
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


def piece_at_t(i, t):
    """3 phases: scatter outward (0..SCATTER_END), hold scattered (..HOLD_END), reassemble (..1)."""
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
    # Rotation runs smoothly across the whole animation
    theta = float(start_locs[i, 2] + smoothstep(t) * theta_delta[i])
    return int(round(pos[1])), int(round(pos[0])), theta


def paste_piece(canvas, piece, x, y, theta):
    layer = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 0))
    layer.paste(piece, (x - PIECE_HALF, y - PIECE_HALF), piece)
    if theta != 0.0:
        layer = layer.rotate(theta, resample=Image.BILINEAR, center=(x, y))
    canvas.alpha_composite(layer)


def make_frame(t, caption=''):
    canvas = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))
    for i in range(16):
        x, y, theta = piece_at_t(i, t)
        paste_piece(canvas, base1[i], x, y, theta)
    canvas_rgb = canvas.convert('RGB')
    draw_caption(canvas_rgb, caption)
    if OUT_SIZE != CANVAS_SIZE:
        canvas_rgb = canvas_rgb.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    return np.array(canvas_rgb)


def static_frame(img, caption=''):
    canvas = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    canvas.paste(Image.fromarray(img), (img_offset, img_offset))
    draw_caption(canvas, caption)
    if OUT_SIZE != CANVAS_SIZE:
        canvas = canvas.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    return np.array(canvas)


def main():
    print(f'Rendering {N_FRAMES} frames '
          f'({HOLD1_FRAMES} hold1 + {ANIM_FRAMES} anim + {HOLD2_FRAMES} hold2) '
          f'at {FPS}fps -> {OUT_PATH}')
    frame1_static = static_frame(img1, caption=args.caption1)
    # Hold the assembled-pieces frame at the end (image1's pieces in image2's
    # arrangement) — that's the actual "puzzle solved" view; image2 itself is
    # only an approximation of it because the model softens the seams.
    final_anim_frame = make_frame(1.0, caption=args.caption2)

    frames = np.empty((N_FRAMES, OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)
    for f in range(N_FRAMES):
        if f < HOLD1_FRAMES:
            frames[f] = frame1_static
        elif f >= HOLD1_FRAMES + ANIM_FRAMES:
            frames[f] = final_anim_frame
        else:
            anim_t = (f - HOLD1_FRAMES) / max(1, ANIM_FRAMES - 1)
            frames[f] = make_frame(smoothstep(anim_t))
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
