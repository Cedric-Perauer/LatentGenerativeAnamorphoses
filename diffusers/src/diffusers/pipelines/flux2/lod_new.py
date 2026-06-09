"""
Laplacian Pyramid Warping implementation based on:
"Taming Rectified Flow for Inversion and Editing" (Appendix A.3-A.4)

This module implements:
- Laplacian/Gaussian pyramid construction and reconstruction
- Forward warping (view) using 3D grid sampling across pyramid levels
- Backward/inverse warping using backpropagation
- Pyramid blending with partial views and detail-preserving averaging
- Jigsaw puzzle permutation for visual anagrams
"""

import torch
import torch.nn.functional as F
import scipy.ndimage
import numpy as np


# =============================================================================
# Jigsaw Puzzle Permutation
# =============================================================================

def get_inv_perm(perm):
    """
    Get the inverse permutation of a permutation.
    perm[perm_inv] = perm_inv[perm] = arange(len(perm))
    """
    perm_inv = torch.empty_like(perm)
    perm_inv[perm] = torch.arange(len(perm))
    return perm_inv


def get_jigsaw_pieces(size):
    """
    Load all 16 jigsaw puzzle piece masks from PNG files.
    
    The pieces are arranged as:
        c0 e0 f0 c1
        f3 i0 i1 e1
        e3 i3 i2 f1
        c3 f2 e2 c2
    
    where c=corner, i=inner, e=edge1, f=edge2
    
    Args:
        size: Image size (64, 256, or 1024)
    
    Returns:
        pieces: numpy array of shape (16, size, size) with binary masks
    """
    from pathlib import Path
    from PIL import Image
    
    # Location of pieces - relative to the diffusers directory
    piece_dir = Path(__file__).parent.parent.parent.parent.parent / 'puzzle_4x4'
    
    def load_pieces(path):
        """Load a piece and return all 4 rotations."""
        piece = Image.open(path)
        piece = np.array(piece)
        # Handle different image formats
        if piece.ndim == 3:
            piece = piece[:, :, 0]
        piece = (piece > 127).astype(np.float32)
        pieces = np.stack([np.rot90(piece, k=-i) for i in range(4)])
        return pieces
    
    # Load pieces and rotate to get 16 pieces total
    pieces_corner = load_pieces(piece_dir / f'4x4_corner_{size}.png')
    pieces_inner = load_pieces(piece_dir / f'4x4_inner_{size}.png')
    pieces_edge1 = load_pieces(piece_dir / f'4x4_edge1_{size}.png')
    pieces_edge2 = load_pieces(piece_dir / f'4x4_edge2_{size}.png')
    
    # Concatenate in order: corner, inner, edge1, edge2
    pieces = np.concatenate([pieces_corner, pieces_inner, pieces_edge1, pieces_edge2])
    
    return pieces


def make_jigsaw_perm(size, seed=42):
    """
    Create a jigsaw puzzle permutation using proper interlocking pieces.
    
    Based on the visual_anagrams paper implementation.
    
    Args:
        size: Image size (64, 256, or 1024)
        seed: Random seed for reproducible permutation
    
    Returns:
        perm: Pixel permutation tensor of shape (size*size,)
        piece_info: Tuple of (piece_perms, edge_swaps)
    """
    np.random.seed(seed)
    
    # Get random permutations for each piece type (groups of 4)
    identity = np.arange(4)
    perm_corner = np.random.permutation(identity)
    perm_inner = np.random.permutation(identity)
    perm_edge1 = np.random.permutation(identity)
    perm_edge2 = np.random.permutation(identity)
    edge_swaps = np.random.randint(2, size=4)
    piece_perms = np.concatenate([perm_corner, perm_inner, perm_edge1, perm_edge2])
    
    # Load the 16 jigsaw piece masks
    pieces = get_jigsaw_pieces(size)
    
    # Build pixel permutation
    perm = []
    
    for y in range(size):
        for x in range(size):
            # Figure out which piece (x, y) is in
            piece_idx = pieces[:, y, x].argmax()
            
            # Figure out rotation index (which rotation of the base piece)
            rot_idx = piece_idx % 4
            
            # Get destination rotation from permutation
            dest_rot_idx = piece_perms[piece_idx]
            angle = (dest_rot_idx - rot_idx) * 90 / 180 * np.pi
            
            # Center coordinates on origin
            cx = x - (size - 1) / 2.0
            cy = y - (size - 1) / 2.0
            
            # Perform rotation
            nx = np.cos(angle) * cx - np.sin(angle) * cy
            ny = np.sin(angle) * cx + np.cos(angle) * cy
            
            # Translate back and round to nearest integer
            nx = nx + (size - 1) / 2.0
            ny = ny + (size - 1) / 2.0
            nx = int(np.rint(nx))
            ny = int(np.rint(ny))
            
            # Clamp to valid range
            nx = max(0, min(size - 1, nx))
            ny = max(0, min(size - 1, ny))
            
            # Perform swap if piece is an edge and swap is enabled
            new_piece_idx = pieces[:, ny, nx].argmax()
            edge_idx = new_piece_idx % 4
            if new_piece_idx >= 8 and edge_swaps[edge_idx] == 1:
                is_f_edge = (new_piece_idx - 8) // 4  # 1 if edge2, 0 if edge1
                edge_type_parity = 1 - 2 * is_f_edge
                rotation_parity = 1 - 2 * (edge_idx // 2)
                swap_dist = size // 4
                
                # Swap in x or y direction based on edge index
                if edge_idx % 2 == 0:
                    nx = nx + swap_dist * edge_type_parity * rotation_parity
                else:
                    ny = ny + swap_dist * edge_type_parity * rotation_parity
                
                # Clamp again - ensure strictly within bounds
                nx = max(0, min(size - 1, nx))
                ny = max(0, min(size - 1, ny))
            
            # Append new index to permutation - ensure within bounds
            new_idx = int(ny * size + nx)
            new_idx = max(0, min(size * size - 1, new_idx))
            perm.append(new_idx)
    
    return torch.tensor(perm), (piece_perms, edge_swaps)


def create_jigsaw_warp(height, width, seed=42, inverse=False):
    """
    Create a warp field for jigsaw puzzle transformation.
    
    Uses proper interlocking jigsaw pieces loaded from mask files.
    Works by creating the permutation at mask resolution, then scaling.
    
    Args:
        height: Image height
        width: Image width (should equal height)
        seed: Random seed for reproducible permutation
        inverse: If True, create the inverse (unshuffling) warp
    
    Returns:
        warp: (1, 3, H, W) warp field with normalized UV coordinates [0, 1]
    """
    assert height == width, "Jigsaw requires square images"
    size = height
    
    # Map to nearest supported size for piece masks
    if size <= 64:
        mask_size = 64
    elif size <= 256:
        mask_size = 256
    else:
        mask_size = 1024
    
    # Get the pixel permutation at mask resolution
    perm, _ = make_jigsaw_perm(mask_size, seed=seed)
    
    # Convert permutation to normalized source coordinates [0, 1]
    # This way we can scale to any target size
    src_y_norm = (perm // mask_size).float() / (mask_size - 1)
    src_x_norm = (perm % mask_size).float() / (mask_size - 1)
    
    # Reshape to 2D at mask resolution
    src_y_norm = src_y_norm.view(mask_size, mask_size)
    src_x_norm = src_x_norm.view(mask_size, mask_size)
    
    # Resize to target size if needed
    if size != mask_size:
        src_y_norm = F.interpolate(
            src_y_norm.unsqueeze(0).unsqueeze(0), 
            size=(size, size), 
            mode='nearest'
        ).squeeze()
        src_x_norm = F.interpolate(
            src_x_norm.unsqueeze(0).unsqueeze(0), 
            size=(size, size), 
            mode='nearest'
        ).squeeze()
    
    if inverse:
        # For inverse, we need to create the inverse mapping
        # The forward warp tells us: for each dest pixel, where to sample from
        # For inverse, we need: for each source pixel, where it goes to
        
        # Discretize the source coordinates
        src_y_idx = (src_y_norm * (size - 1)).round().long().clamp(0, size - 1)
        src_x_idx = (src_x_norm * (size - 1)).round().long().clamp(0, size - 1)
        
        # Create destination coordinate grids (normalized)
        dy_grid = torch.arange(size).float().view(-1, 1).expand(size, size) / (size - 1)
        dx_grid = torch.arange(size).float().view(1, -1).expand(size, size) / (size - 1)
        
        # Flatten for scatter
        src_flat = src_y_idx.flatten() * size + src_x_idx.flatten()
        
        # Create inverse mapping using scatter
        inv_y = torch.zeros(size * size)
        inv_x = torch.zeros(size * size)
        
        inv_y.scatter_(0, src_flat, dy_grid.flatten())
        inv_x.scatter_(0, src_flat, dx_grid.flatten())
        
        src_y_norm = inv_y.view(size, size)
        src_x_norm = inv_x.view(size, size)
    
    # Clamp to valid range
    src_y_norm = src_y_norm.clamp(0, 1)
    src_x_norm = src_x_norm.clamp(0, 1)
    
    warp = torch.stack([src_x_norm, src_y_norm], dim=0).unsqueeze(0)  # (1, 2, H, W)
    third_channel = torch.zeros(1, 1, size, size)
    warp = torch.cat([warp, third_channel], dim=1)  # (1, 3, H, W)
    
    return warp


# =============================================================================
# Warp Field Creation
# =============================================================================

def create_identity_warp(height=1024, width=1024):
    """
    Create an identity warp field that doesn't transform the image.
    Returns: warp field of shape (1, 3, H, W) with normalized UV coordinates [0, 1]
    """
    v_coords = torch.linspace(0, 1, height).view(-1, 1).expand(height, width)
    u_coords = torch.linspace(0, 1, width).view(1, -1).expand(height, width)
    
    warp = torch.stack([u_coords, v_coords], dim=0).unsqueeze(0)  # (1, 2, H, W)
    third_channel = torch.zeros(1, 1, height, width)
    warp = torch.cat([warp, third_channel], dim=1)  # (1, 3, H, W)
    
    return warp


def create_vertical_flip_warp(height=1024, width=1024):
    """
    Create a vertical flip warp field that flips the image upside down.
    Returns: warp field of shape (1, 3, H, W) with normalized UV coordinates [0, 1]
    """
    v_coords = torch.linspace(1, 0, height).view(-1, 1).expand(height, width)  # Flipped
    u_coords = torch.linspace(0, 1, width).view(1, -1).expand(height, width)
    
    warp = torch.stack([u_coords, v_coords], dim=0).unsqueeze(0)  # (1, 2, H, W)
    third_channel = torch.zeros(1, 1, height, width)
    warp = torch.cat([warp, third_channel], dim=1)  # (1, 3, H, W)
    
    return warp


def create_circular_mask(height, width, radius_ratio=0.45, feather_ratio=0.05):
    """
    Create a circular mask centered on the image with soft feathered edges.
    
    Args:
        height: Image height
        width: Image width
        radius_ratio: Radius as a fraction of min(height, width) / 2
                      Default 0.45 gives a circle that almost fills the image
        feather_ratio: Width of the soft edge as fraction of radius
                       Default 0.05 gives a subtle gradient at the edge
    
    Returns:
        mask: (1, 1, H, W) tensor with 1.0 inside circle, smooth falloff at edge, 0.0 outside
    """
    cy, cx = height / 2, width / 2
    radius = radius_ratio * min(height, width)
    feather_width = feather_ratio * radius
    
    y = torch.arange(height).float().view(-1, 1).expand(height, width)
    x = torch.arange(width).float().view(1, -1).expand(height, width)
    
    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # Smooth falloff: 1 inside (radius - feather), 0 outside (radius + feather)
    # Linear interpolation in the feather zone
    inner_radius = radius - feather_width
    outer_radius = radius + feather_width
    
    mask = torch.clamp((outer_radius - dist) / (2 * feather_width + 1e-6), 0.0, 1.0)
    
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def create_circular_rotation_warp(height, width, angle_degrees, radius_ratio=0.45, feather_ratio=0.08):
    """
    Create a warp field that rotates a circular region by the given angle.
    Pixels outside the circle map to themselves (identity).
    The warp itself is applied fully inside the circle, but the returned mask
    has soft edges for smooth blending.
    
    Args:
        height: Image height
        width: Image width
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
        radius_ratio: Radius as fraction of min(height, width) / 2
        feather_ratio: Width of soft edge as fraction of radius (default 0.08)
    
    Returns:
        warp: (1, 3, H, W) warp field with normalized UV coordinates [0, 1]
        mask: (1, 1, H, W) circular mask with soft feathered edges
    """
    cy, cx = height / 2, width / 2
    radius = radius_ratio * min(height, width)
    feather_width = feather_ratio * radius
    
    # Create coordinate grids
    y = torch.arange(height).float().view(-1, 1).expand(height, width)
    x = torch.arange(width).float().view(1, -1).expand(height, width)
    
    # Distance from center
    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # For warping, use the outer edge of feather zone to include the blend region
    warp_radius = radius + feather_width
    inside_warp_zone = dist <= warp_radius
    
    # Convert to radians
    angle_rad = angle_degrees * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # For each destination pixel, find the source pixel
    # Rotation around center: apply inverse rotation to find source
    x_rel = x - cx
    y_rel = y - cy
    
    x_src = cos_a * x_rel + sin_a * y_rel + cx
    y_src = -sin_a * x_rel + cos_a * y_rel + cy
    
    # Normalize to [0, 1]
    u_coords = x / (width - 1)  # Identity mapping
    v_coords = y / (height - 1)
    
    u_src = x_src / (width - 1)  # Rotated mapping
    v_src = y_src / (height - 1)
    
    # Apply rotation in the warp zone (slightly larger than mask for clean edges)
    warp_u = torch.where(inside_warp_zone, u_src, u_coords)
    warp_v = torch.where(inside_warp_zone, v_src, v_coords)
    
    warp = torch.stack([warp_u, warp_v], dim=0).unsqueeze(0)  # (1, 2, H, W)
    third_channel = torch.zeros(1, 1, height, width)
    warp = torch.cat([warp, third_channel], dim=1)  # (1, 3, H, W)
    
    # Create soft-edged mask for blending
    inner_radius = radius - feather_width
    outer_radius = radius + feather_width
    mask = torch.clamp((outer_radius - dist) / (2 * feather_width + 1e-6), 0.0, 1.0)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return warp, mask


def _raytrace_conic_mirror(
    height, width,
    cone_height=1.0,
    cone_radius=1.0,
    cone_half_angle_deg=35.0,
    eye_dir=(0.0, 0.0, -1.0),
    table_extent=3.0,
):
    """Compute the conic-mirror UV map by direct ray reflection on the cone.

    Setup (matches LookingGlass paper supplementary panel (c)):
      - Apex-up cone at the canvas center. Cone half-angle defaults to 22°
        (slightly steeper than 45°) so that *every* reflected ray has a
        downward component and lands on the ground — i.e. the warp is
        well-defined over the whole disk, not just one half. (A 45° cone
        with oblique illumination has half its surface reflecting upward
        and never hitting the ground; smaller angles avoid that.)
      - Painting (rect mirror) on z = 0, occupying [-table_extent, table_extent]².
      - Each canvas pixel (px, py) in the disk maps directly to the cone
        surface point at (px, py, Hc − r/tan α). No perspective camera —
        we trace illumination from a parallel direction ``eye_dir`` so
        every cone surface point is mapped (no self-occlusion holes).

    Returns
    -------
    u_src, v_src : (H, W) float32 — source UVs in [0, 1].
    valid        : (H, W) bool   — True inside the disk and where the
                                   reflection lands on the rect.
    """
    Hc = float(cone_height)
    alpha = np.deg2rad(cone_half_angle_deg)
    tan_a = np.tan(alpha)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    # Cone radius at the base (z = 0) given height Hc and half-angle.
    base_radius = Hc * tan_a
    # Canvas radius in scene units = base_radius (the disk silhouette).
    canvas_radius_scene = base_radius

    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    ys_grid, xs_grid = np.indices((height, width)).astype(np.float64)
    px = (xs_grid - cx) / (min(height, width) / 2.0) * canvas_radius_scene
    py = (ys_grid - cy) / (min(height, width) / 2.0) * canvas_radius_scene
    r_xy = np.sqrt(px ** 2 + py ** 2)
    inside_disk = r_xy <= canvas_radius_scene

    # Cone surface: r = tan α (H − z) → z(r) = H − r / tan α.
    surf_z = Hc - r_xy / max(tan_a, 1e-12)
    # Outward normal at half-angle α: n = (cos α cos θ, cos α sin θ, sin α).
    safe_r = np.where(r_xy > 1e-9, r_xy, 1e-9)
    cos_th = px / safe_r
    sin_th = py / safe_r
    nx = cos_a * cos_th
    ny = cos_a * sin_th
    nz = np.full_like(px, sin_a)

    dx0, dy0, dz0 = (np.array(eye_dir, dtype=np.float64)
                     / (np.linalg.norm(eye_dir) + 1e-12))
    d_dot_n = dx0 * nx + dy0 * ny + dz0 * nz
    rx = dx0 - 2.0 * d_dot_n * nx
    ry = dy0 - 2.0 * d_dot_n * ny
    rz = dz0 - 2.0 * d_dot_n * nz

    t_g = -surf_z / (rz - 1e-12)
    gx = px + t_g * rx
    gy = py + t_g * ry
    hits_ground = (rz < 0) & (t_g > 0)

    u_src = (gx + table_extent) / (2.0 * table_extent)
    v_src = (gy + table_extent) / (2.0 * table_extent)
    in_rect = (u_src >= 0) & (u_src <= 1) & (v_src >= 0) & (v_src <= 1)
    valid = inside_disk & hits_ground & in_rect

    return u_src.astype(np.float32), v_src.astype(np.float32), valid


def create_conic_mirror_warp(
    height,
    width,
    r_in_ratio=0.0,
    r_out_ratio=0.95,
    feather_ratio=0.04,
    inverse=False,
    eye_dir=(0.0, 0.0, -1.0),
    cone_height=1.0,
    cone_half_angle_deg=35.0,
    table_extent=3.0,
):
    """
    Conic mirror anamorphosis warp (polar unwrap of an annulus).

    Geometry: an apex-up 45° cone sits at the canvas center. The painted
    image lives on the table in the annulus r in [r_in, r_out] around the
    center; the viewer looks straight down. The perceived (rectified) image
    is the polar unwrap of that annulus, where the horizontal axis encodes
    angle theta in [-pi, pi) and the vertical axis encodes radius (top of
    the rectangle = outer edge of the annulus, bottom = inner edge near the
    cone apex). This matches the standard cv2.warpPolar(WARP_POLAR_LINEAR)
    and the Looking Glass paper convention.

    Convention (matches the rotation/jigsaw warps in this module):
      - inverse=False: destination is the canonical (annular table) view,
        source is the rectangular mirror view; outside the annulus the
        warp is identity (so the surrounding canvas passes through).
      - inverse=True : destination is the rectangular mirror view, source
        is the canonical annular view.

    Args:
        height, width: square canvas dims (H == W expected).
        r_in_ratio:  inner annulus radius as a fraction of min(H, W) / 2.
        r_out_ratio: outer annulus radius as a fraction of min(H, W) / 2.
        feather_ratio: width of the soft annulus boundary as a fraction
            of (r_out - r_in). Used only for the forward (annular) mask.
        inverse: see above.

    Returns:
        warp: (1, 3, H, W) UV warp field with values in [0, 1].
        mask: (1, 1, H, W) blending mask in [0, 1]. Annular for the forward
            direction, all-ones for the inverse direction.
    """
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    r_max = min(height, width) / 2.0
    feather = max(1.0, feather_ratio * r_max * r_out_ratio)
    canvas_radius = r_out_ratio  # canvas disk radius in scene units

    y = torch.arange(height).float().view(-1, 1).expand(height, width)
    x = torch.arange(width).float().view(1, -1).expand(height, width)
    dx = x - cx
    dy = y - cy
    r_pix = torch.sqrt(dx * dx + dy * dy)

    # Ray-trace the conic mirror to produce (u_src, v_src) and a validity
    # mask for the forward direction (canvas → rect mirror via cone).
    u_fwd_np, v_fwd_np, valid_fwd_np = _raytrace_conic_mirror(
        height, width,
        cone_height=cone_height,
        cone_radius=cone_height,
        cone_half_angle_deg=cone_half_angle_deg,
        eye_dir=eye_dir,
        table_extent=table_extent,
    )
    u_fwd = torch.from_numpy(u_fwd_np)
    v_fwd = torch.from_numpy(v_fwd_np)
    valid_fwd = torch.from_numpy(valid_fwd_np)

    if not inverse:
        # Forward: dest = canvas (canonical), source = rect mirror (rectified).
        # Holes inside the disk silhouette (where the ray-trace hit the back
        # of the cone or the reflection escaped the rect) are filled with a
        # multi-scale Gaussian "push-pull" fill — strictly better than the
        # axis-aligned roll/box-blur fill, which produced visible diagonal
        # streaks because values only propagated row/column-wise.
        u_np = u_fwd.cpu().numpy().astype(np.float32)
        v_np = v_fwd.cpu().numpy().astype(np.float32)
        w_np = valid_fwd.cpu().numpy().astype(np.float32)
        for sigma in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
            if w_np.min() > 0.999:
                break
            wu = scipy.ndimage.gaussian_filter(u_np * w_np, sigma=sigma)
            wv = scipy.ndimage.gaussian_filter(v_np * w_np, sigma=sigma)
            ww = scipy.ndimage.gaussian_filter(w_np,        sigma=sigma)
            new_data = (w_np < 0.5) & (ww > 1e-4)
            u_np = np.where(new_data, wu / (ww + 1e-8), u_np)
            v_np = np.where(new_data, wv / (ww + 1e-8), v_np)
            w_np = np.where(new_data, np.ones_like(w_np), w_np)
        u_filled = torch.from_numpy(u_np)
        v_filled = torch.from_numpy(v_np)
        has = torch.from_numpy(w_np > 0.5)

        u_id = x / (width - 1)
        v_id = y / (height - 1)
        # Use ray-traced UV everywhere inside the disk silhouette (filled),
        # identity outside.
        inside_disk = r_pix <= (r_max * r_out_ratio)
        warp_u = torch.where(inside_disk & has, u_filled, u_id)
        warp_v = torch.where(inside_disk & has, v_filled, v_id)
        warp_u = torch.clamp(warp_u, 0.0, 1.0)
        warp_v = torch.clamp(warp_v, 0.0, 1.0)

        # Strict binary disk mask — no feather. The lwp() path hard-composites
        # with this mask so the cone silhouette has a sharp boundary in the
        # final blend (no canoe content bleeding past the disk into pred1).
        mask = inside_disk.float().unsqueeze(0).unsqueeze(0)
    else:
        # Inverse: dest = rect mirror, source = canvas (canonical).
        # Build the inverse by scattering the forward UVs.
        u_inv = torch.zeros(height, width, dtype=torch.float32)
        v_inv = torch.zeros(height, width, dtype=torch.float32)
        count = torch.zeros(height, width, dtype=torch.float32)

        x_canon = (x / (width - 1)).flatten()
        y_canon = (y / (height - 1)).flatten()
        u_dst = (u_fwd.flatten() * (width - 1)).round().long().clamp(0, width - 1)
        v_dst = (v_fwd.flatten() * (height - 1)).round().long().clamp(0, height - 1)
        valid_flat = valid_fwd.flatten()
        idx = (v_dst * width + u_dst)[valid_flat]
        u_inv.flatten().scatter_add_(0, idx, x_canon[valid_flat])
        v_inv.flatten().scatter_add_(0, idx, y_canon[valid_flat])
        count.flatten().scatter_add_(0, idx, torch.ones_like(x_canon)[valid_flat])

        has_data = count > 0
        u_inv = torch.where(has_data, u_inv / count.clamp(min=1.0), torch.zeros_like(u_inv))
        v_inv = torch.where(has_data, v_inv / count.clamp(min=1.0), torch.zeros_like(v_inv))

        # The mask encodes three regions in one float channel so callers
        # can recover them without re-running the scatter:
        #   1.0  → natively valid (real scatter contributions)
        #   0.5  → apex hole — filled by interpolation, blur recommended
        #   0.0  → outside the disk silhouette (strict black)
        # ``mask > 0.5`` (used by the pipeline's lwp) recovers the full
        # active region; the test script reads the float values to detect
        # the apex hole and apply targeted blur there.
        has_data_np = has_data.cpu().numpy()
        filled_np = scipy.ndimage.binary_fill_holes(has_data_np)
        apex_hole_np = filled_np & (~has_data_np)
        encoded_np = (has_data_np.astype(np.float32) * 1.0
                      + apex_hole_np.astype(np.float32) * 0.5)
        inverse_mask_2d = torch.from_numpy(encoded_np)

        # Multi-scale Gaussian push-pull fill (pyramid-style). Eliminates
        # axis-aligned streaks that a box-blur or roll-based fill produces:
        # at each scale we form a Gaussian-weighted average sum(u·w)/sum(w)
        # and use it to fill cells whose hole is wider than the current
        # scale. Successively coarser scales pull values across the apex
        # hole (the rect-mirror centre) without any directional bias.
        u_np = u_inv.cpu().numpy().astype(np.float32)
        v_np = v_inv.cpu().numpy().astype(np.float32)
        w_np = has_data.cpu().numpy().astype(np.float32)
        for sigma in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
            if w_np.min() > 0.999:
                break
            wu = scipy.ndimage.gaussian_filter(u_np * w_np, sigma=sigma)
            wv = scipy.ndimage.gaussian_filter(v_np * w_np, sigma=sigma)
            ww = scipy.ndimage.gaussian_filter(w_np,        sigma=sigma)
            new_known = (w_np < 0.5) & (ww > 1e-4)
            u_np = np.where(new_known, wu / (ww + 1e-8), u_np)
            v_np = np.where(new_known, wv / (ww + 1e-8), v_np)
            w_np = np.where(new_known, np.ones_like(w_np), w_np)
        u_filled = torch.from_numpy(u_np)
        v_filled = torch.from_numpy(v_np)

        warp_u = torch.clamp(u_filled, 0.0, 1.0)
        warp_v = torch.clamp(v_filled, 0.0, 1.0)
        mask = inverse_mask_2d.float().unsqueeze(0).unsqueeze(0)

    warp = torch.stack([warp_u, warp_v], dim=0).unsqueeze(0)  # (1, 2, H, W)
    third_channel = torch.zeros(1, 1, height, width)
    warp = torch.cat([warp, third_channel], dim=1)            # (1, 3, H, W)

    return warp, mask


def create_conic_inner_mirror_warp(
    height,
    width,
    radius_ratio=0.27,
    feather_ratio=0.08,
    inverse=False,
):
    """Conic-mirror anamorphosis: an inner circle of view 1 hides the
    WHOLE of view 2.

    View 1 (the canonical painting) is modified only inside the circle of
    radius R = radius_ratio * min(H, W). View 2 (what the cone mirror
    shows) is a full image of its own — as in the LookingGlass paper,
    where the mirror view fills the camera frame. The two are related by
    the cone's radial inversion plus a scale: a view-2 pixel at polar
    coords (rho, theta) corresponds to the view-1 circle point at

        r(rho, theta) = R * (1 - rho / edge(theta)),

    where edge is the inscribed-circle radius of view 2 — view 2's
    center shows the circle rim and
    view 2's boundary shows the circle center (the conic mirror's
    center <-> rim inversion), with the full square canvas of view 2
    compressed into the circle. Outside the circle, view 1 passes through
    untouched.

    Constraining the whole of view 2 (rather than only its central disk)
    is what keeps the second prompt effective when the circle is small:
    every pixel of the view-2 prediction participates in the blend.

    Directions (matching the other warps in this module):
      - inverse=False: dest = view-1/print space. Circle pixels sample
        view 2 at rho = edge(theta) * (1 - r / R); identity outside.
        mask = feathered disk (the blend region).
      - inverse=True:  dest = view-2/mirror space. Every pixel samples the
        view-1 circle at r(rho, theta). mask = all-ones (view 2 is fully
        defined; no confinement).

    Returns:
        warp: (1, 3, H, W) UV warp field with values in [0, 1].
        mask: (1, 1, H, W) blend mask in [0, 1].
    """
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    R = radius_ratio * min(height, width)
    feather = max(1.0, feather_ratio * R)

    y = torch.arange(height).float().view(-1, 1).expand(height, width)
    x = torch.arange(width).float().view(1, -1).expand(height, width)
    dx = x - cx
    dy = y - cy
    r = torch.sqrt(dx * dx + dy * dy)
    safe_r = torch.clamp(r, min=1e-6)

    # Radial reach in view-2 space: the inscribed-circle radius. A
    # constant reach keeps the map rotationally symmetric — following the
    # square boundary instead (edge(theta)) puts derivative kinks along
    # the diagonals that show up as an X-shaped seam in both views.
    reach = 0.5 * min(height, width)

    if inverse:
        # View 2 samples the view-1 circle: r(rho) = R * (1 - rho / reach).
        # Pixels beyond the inscribed circle (the corners) clamp to the
        # reach boundary, extending the rim content radially outward.
        r_src = R * (1.0 - torch.clamp(r, max=reach) / reach)
        scale = r_src / safe_r
        region = torch.ones_like(r, dtype=torch.bool)
        mask = torch.ones(height, width)
    else:
        # The view-1 circle samples view 2: rho(r) = reach * (1 - r / R).
        rho = reach * (1.0 - r / R)
        scale = rho.clamp(min=0.0) / safe_r
        region = r <= R + feather
        # Soft disk mask: 1 inside, 0 outside, linear ramp across a
        # 2*feather band centered on the rim (same convention as
        # create_circular_rotation_warp). lwp()'s blend and the pipeline's
        # confinement composite both key off this.
        mask = torch.clamp(((R - r) + feather) / (2.0 * feather), 0.0, 1.0)

    x_src = cx + dx * scale
    y_src = cy + dy * scale
    u_id = x / (width - 1)
    v_id = y / (height - 1)
    u_src = (x_src / (width - 1)).clamp(0.0, 1.0)
    v_src = (y_src / (height - 1)).clamp(0.0, 1.0)
    warp_u = torch.where(region, u_src, u_id)
    warp_v = torch.where(region, v_src, v_id)

    warp = torch.stack([warp_u, warp_v], dim=0).unsqueeze(0)  # (1, 2, H, W)
    third_channel = torch.zeros(1, 1, height, width)
    warp = torch.cat([warp, third_channel], dim=1)            # (1, 3, H, W)
    return warp, mask.unsqueeze(0).unsqueeze(0)


# =============================================================================
# LOD Level Computation
# =============================================================================

def _compute_lod_level(uv_map: torch.Tensor, maxLOD: int) -> torch.Tensor:
    """
    Compute LOD level per pixel based on the Jacobian of the UV mapping.
    
    Args:
        uv_map: (B, 2, H, W) tensor of UV coordinates scaled to image dimensions
        maxLOD: maximum LOD level
    Returns:
        lod: (H, W) tensor of LOD levels
    """
    # uv_map is (B, 2, H, W), convert to (H, W, 2) for gradient computation
    uv = uv_map[0].permute(1, 2, 0)  # (H, W, 2)
    u = uv[..., 0]
    v = uv[..., 1]
    
    # Compute gradients (central difference)
    u_x = F.pad(u[2:, :] - u[:-2, :], (0, 0, 1, 1)) / 2
    u_y = F.pad(u[:, 2:] - u[:, :-2], (1, 1, 0, 0)) / 2
    v_x = F.pad(v[2:, :] - v[:-2, :], (0, 0, 1, 1)) / 2
    v_y = F.pad(v[:, 2:] - v[:, :-2], (1, 1, 0, 0)) / 2
    
    # Norm of the Jacobian
    jac_norm = torch.sqrt(u_x ** 2 + u_y ** 2 + v_x ** 2 + v_y ** 2)
    jac_norm = torch.clamp(jac_norm, min=1e-6)
    
    lod = 0.85 * torch.log2(jac_norm)
    lod = torch.clamp(lod, min=0.0, max=float(maxLOD))
    return lod


# =============================================================================
# Pyramid Operations
# =============================================================================

_BURT_KERNEL_1D = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def pyrDown(x):
    """Downsample by factor of 2 with a 5-tap Gaussian prefilter (the
    Burt-Adelson kernel [1, 4, 6, 4, 1] / 16, applied separably with
    reflect padding) followed by stride-2 subsampling.

    The Gaussian prefilter matters for the LOD-aware warps: a plain 2x2
    box average aliases, and when a radially-monotone LOD map samples
    across the stacked pyramid levels that aliasing shows up as concentric
    banding in the warped output. Inputs containing NaNs (partial views)
    fall back to the NaN-aware 2x2 averaging, where a Gaussian convolution
    would smear NaNs into valid pixels.
    """
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True

    b, c, h, w = x.shape
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    x = x[:, :, :h_even, :w_even]

    if torch.isnan(x).any():
        # NaN-aware 2x2 block average (legacy path for partial views).
        x_reshaped = x.view(b, c, h_even // 2, 2, w_even // 2, 2)
        x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5)
        x_blocks = x_permuted.reshape(b, c, h_even // 2, w_even // 2, 4)
        out = torch.nanmean(x_blocks, dim=-1)
    else:
        k = _BURT_KERNEL_1D.to(device=x.device, dtype=x.dtype)
        kx = k.view(1, 1, 1, 5).expand(c, 1, 1, 5)
        ky = k.view(1, 1, 5, 1).expand(c, 1, 5, 1)
        out = F.conv2d(F.pad(x, (2, 2, 0, 0), mode='reflect'), kx, groups=c)
        out = F.conv2d(F.pad(out, (0, 0, 2, 2), mode='reflect'), ky, groups=c)
        out = out[:, :, ::2, ::2]

    if squeeze:
        out = out.squeeze(0)
    return out


def pyrUp(x):
    """Upsample by factor of 2 using bilinear interpolation."""
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    
    out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    
    if squeeze:
        out = out.squeeze(0)
    return out


def LaplacianPyramid(x, leveln):
    """
    Build a Laplacian pyramid from image x.
    
    Args:
        x: Input image tensor (B, C, H, W) or (C, H, W)
        leveln: Number of pyramid levels
    
    Returns:
        List of Laplacian pyramid levels. Each level (except last) contains
        high-frequency details. The last level is the low-res Gaussian base.
    """
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    
    # Build Gaussian pyramid
    gp = [x]
    for i in range(leveln - 1):
        x = pyrDown(x)
        gp.append(x)
    
    # Build Laplacian pyramid from Gaussian pyramid
    lp = []
    for i in range(leveln - 1):
        upsampled = pyrUp(gp[i + 1])
        # Handle size mismatch
        if upsampled.shape != gp[i].shape:
            upsampled = F.interpolate(upsampled, size=gp[i].shape[-2:], mode='bilinear', align_corners=False)
        L = gp[i] - upsampled
        lp.append(L)
    lp.append(gp[-1])  # Last level is the smallest Gaussian
    
    if squeeze:
        lp = [l.squeeze(0) for l in lp]
    
    return lp


def Laplacian2Gaussian(lp):
    """
    Reconstruct image from Laplacian pyramid.
    
    Args:
        lp: Laplacian pyramid (list of tensors)
    
    Returns:
        List representing Gaussian pyramid, with gp[0] being the full resolution image.
    """
    gp = [lp[-1]]  # Start with the coarsest level
    
    for i in range(len(lp) - 2, -1, -1):
        x = gp[-1]
        squeeze = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True
        
        upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        if squeeze:
            upsampled = upsampled.squeeze(0)
        
        # Handle size mismatch
        target = lp[i]
        if upsampled.shape != target.shape:
            if upsampled.ndim == 3:
                upsampled = upsampled.unsqueeze(0)
                upsampled = F.interpolate(upsampled, size=target.shape[-2:], mode='bilinear', align_corners=False)
                upsampled = upsampled.squeeze(0)
            else:
                upsampled = F.interpolate(upsampled, size=target.shape[-2:], mode='bilinear', align_corners=False)
        
        G = target + upsampled
        gp.append(G)
    
    gp.reverse()
    return gp


def pyrStack(pyramid, dim=-1):
    """
    Upsample all pyramid levels to highest resolution and stack along dim.
    
    Args:
        pyramid: List of pyramid levels (from coarse to fine or fine to coarse)
        dim: Dimension to stack along
    
    Returns:
        Stacked tensor with all levels at full resolution
    """
    # Find the largest size (should be the first or last level)
    sizes = [p.shape[-2:] for p in pyramid]
    target_size = max(sizes, key=lambda s: s[0] * s[1])
    
    upsampled = []
    for level in pyramid:
        squeeze = False
        if level.ndim == 3:
            level = level.unsqueeze(0)
            squeeze = True
        
        if level.shape[-2:] != target_size:
            level = F.interpolate(level, size=target_size, mode='bilinear', align_corners=True)
        
        if squeeze:
            level = level.squeeze(0)
        
        upsampled.append(level)
    
    return torch.stack(upsampled, dim=dim)


# =============================================================================
# Grid Construction
# =============================================================================

def _get_grid(warp, maxLOD):
    """
    Convert warp field to 3D grid for grid_sample across pyramid levels.
    
    Args:
        warp: Warp field of shape (1, 3, H, W) with UV in [0, 1]
        maxLOD: Maximum LOD level
    
    Returns:
        grid: (1, 1, H, W, 3) tensor with (u, v, lod) coordinates in [-1, 1].
        Order matches F.grid_sample's 5D convention (x→W, y→H, z→D), where
        the layers tensor is stacked along D = pyramid level.
    """
    h, w = warp.shape[-2:]

    # Compute LOD level and normalize to (-1, 1)
    mapping = max(h, w) * warp[:, :2]
    lod = _compute_lod_max_col(mapping, maxLOD=maxLOD)
    lod = 2 * lod / max(maxLOD, 1) - 1.0
    lod = lod.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

    # Normalize UV to (-1, 1)
    grid = warp[:, :2].permute(0, 2, 3, 1)  # (1, H, W, 2) order (u, v)

    # Mark undefined regions (UV exactly 0) with NaN.
    nan_mask = torch.all(grid == 0.0, dim=-1, keepdim=True).expand(-1, -1, -1, 2)
    grid = grid.clone()
    grid[nan_mask] = float('nan')
    grid = 2 * grid - 1

    # 5D grid_sample expects last-dim order (x, y, z) with x→W, y→H, z→D.
    # Our layers volume is stacked along D = pyramid level, H = height,
    # W = width, so we want (u, v, lod) here.
    grid = torch.cat([grid, lod], dim=-1).unsqueeze(1)  # (1, 1, H, W, 3)

    return grid


# =============================================================================
# Imputation
# =============================================================================

def impute_with_nearest(img, mask):
    """
    Fill missing values (where mask is True) with nearest valid values.
    
    Args:
        img: Tensor of shape (C, H, W) or (B, C, H, W)
        mask: Boolean tensor (True for valid pixels, False for missing)
    
    Returns:
        Tensor with missing values filled
    """
    squeeze = False
    if img.ndim == 3:
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeeze = True
    
    img_np = img.cpu().numpy()
    mask_np = (~mask).cpu().numpy()  # True where missing
    
    result = np.copy(img_np)
    
    for b in range(img_np.shape[0]):
        for c in range(img_np.shape[1]):
            if np.all(mask_np[b, c]):
                continue
            # Distance transform gives indices of nearest valid pixel
            _, indices = scipy.ndimage.distance_transform_edt(mask_np[b, c], return_indices=True)
            filled = img_np[b, c][indices[0], indices[1]]
            result[b, c][mask_np[b, c]] = filled[mask_np[b, c]]
    
    result = torch.from_numpy(result).to(img.device, dtype=img.dtype)
    
    if squeeze:
        result = result.squeeze(0)
    
    return result


# =============================================================================
# Forward Warping (View) - Simple Version
# =============================================================================

def view_simple(image, warp):
    """
    Simple forward warp using 2D grid sampling.
    This is for transforms like flips where we don't need LOD-based sampling.
    
    Args:
        image: Image tensor (B, C, H, W) or (C, H, W)
        warp: Warp field of shape (1, 3, H, W) with UV in [0, 1]
    
    Returns:
        Warped image tensor (same shape as input)
    """
    squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True
    
    device = image.device
    dtype = image.dtype
    warp = warp.to(device=device, dtype=torch.float32)
    
    # Convert warp from [0,1] to [-1,1] for grid_sample
    # warp[:, 0] is u (horizontal), warp[:, 1] is v (vertical)
    grid = warp[:, :2].permute(0, 2, 3, 1)  # (1, H, W, 2)
    grid = 2 * grid - 1  # [0,1] -> [-1,1]
    
    # Expand grid to match batch size
    if grid.shape[0] != image.shape[0]:
        grid = grid.expand(image.shape[0], -1, -1, -1)
    
    # Apply warp
    warped = F.grid_sample(
        image.float(),
        grid.float(),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    
    if squeeze:
        warped = warped.squeeze(0)
    
    return warped.to(dtype)


def _compute_lod_max_col(uv_map, maxLOD):
    """
    LOD per pixel from the Jacobian's max column norm.

    Unlike ``_compute_lod_level`` (which uses Frobenius norm and a 0.85
    fudge factor and so returns lod ≈ 0.4 for an identity warp), this
    returns lod = 0 when the warp is locally non-contracting. Specifically,
    let s be the larger of the column norms of the Jacobian (i.e. the
    pixel-space scale factor in the dominant direction); then
    lod = clamp(log2(max(s, 1)), 0, maxLOD). At identity s = 1 → lod = 0;
    at 2x compression s = 2 → lod = 1; etc.
    """
    uv = uv_map[0].permute(1, 2, 0)  # (H, W, 2)
    u = uv[..., 0]
    v = uv[..., 1]
    period = float(uv_map.shape[-1])  # u/v are in pixel units of width

    def _wrap(d):
        # Shortest signed difference modulo `period`. Wrap BEFORE the /2 so
        # the conic atan2 seam (raw jump ≈ ±period) collapses to ~0 instead
        # of producing a spurious max-LOD ridge along the θ = ±π line.
        return d - period * torch.round(d / period)

    du_dy = _wrap(F.pad(u[2:, :] - u[:-2, :], (0, 0, 1, 1))) / 2.0
    du_dx = _wrap(F.pad(u[:, 2:] - u[:, :-2], (1, 1, 0, 0))) / 2.0
    dv_dy = _wrap(F.pad(v[2:, :] - v[:-2, :], (0, 0, 1, 1))) / 2.0
    dv_dx = _wrap(F.pad(v[:, 2:] - v[:, :-2], (1, 1, 0, 0))) / 2.0
    col_x = torch.sqrt(du_dx ** 2 + dv_dx ** 2)
    col_y = torch.sqrt(du_dy ** 2 + dv_dy ** 2)
    scale = torch.clamp(torch.maximum(col_x, col_y), min=1.0)
    lod = torch.log2(scale)
    return torch.clamp(lod, min=0.0, max=float(maxLOD))


def soften_inverse_conic(recovered, mask, apex_sigma=60.0,
                         outside_blur_sigma=20.0, outside_fade_dist=40.0,
                         edge_blend_sigma=6.0, value_range=(0.0, 1.0),
                         apex_dilate_frac=0.06):
    """Post-process the conic-mirror inverse-warp result so it matches the
    benchmark inverse_correct.png style:

    - Inside the disk silhouette, the **apex-hole region** (where the
      forward warp had no scatter contributions and the inverse warp
      filled by interpolation) is replaced with a heavy Gaussian blur,
      blended in via a soft weight derived from the apex-hole indicator.
      The mask returned by ``create_conic_mirror_warp(inverse=True)``
      encodes this region as values around 0.5; values around 1.0 are
      native scatter (kept sharp); 0.0 is outside the disk.
    - Outside the disk silhouette, a heavily-blurred copy of the
      recovered content is alpha-faded to black over a short distance,
      so the boundary has a soft halo rather than a hard cutoff.

    Args:
        recovered: (B, C, H, W) torch tensor — output of view_*.
        mask:      (1, 1, H, W) torch tensor — encoded inverse mask from
                   ``create_conic_mirror_warp(inverse=True)``.
        value_range: (low, high) value range of ``recovered``. The pipeline
                   calls this on VAE-decode-space images in [-1, 1]; the
                   test scripts use [0, 1]. All blending happens in a
                   normalized [0, 1] domain (so "fade to black" fades to
                   ``low``) and the result is mapped back to the input range.

    Returns:
        (B, C, H, W) tensor with the same dtype as ``recovered``.
    """
    out_dtype = recovered.dtype
    img_b = recovered.float()
    if img_b.ndim == 3:
        img_b = img_b.unsqueeze(0)
    B, C, H, W = img_b.shape

    lo, hi = float(value_range[0]), float(value_range[1])
    img_b = (img_b - lo) / (hi - lo)

    img_np_b = img_b.clamp(0, 1).float().cpu().numpy()  # (B, C, H, W)
    m = mask[0, 0].clamp(0, 1).float().cpu().numpy().astype(np.float32)

    # Heavy-blur passes for the apex blur and the outside fade. We blur
    # each batch/channel separately along H, W only.
    img_apex = np.empty_like(img_np_b)
    img_out  = np.empty_like(img_np_b)
    for b in range(B):
        for c in range(C):
            img_apex[b, c] = scipy.ndimage.gaussian_filter(img_np_b[b, c], sigma=apex_sigma)
            img_out[b,  c] = scipy.ndimage.gaussian_filter(img_np_b[b, c], sigma=outside_blur_sigma)

    # Apex weight: take the apex-hole region directly from the encoded mask.
    # The large connected hole(s) — the true apex hole under the cone — get
    # dilated outward, because the inverse warp's scatter density collapses
    # in a band AROUND the hole (extreme radial stretching) leaving radial
    # streaks the blur must also cover. Small speckle holes scattered over
    # the disk (isolated scatter misses) are left undilated; blanketing
    # them would blur the entire recovered image.
    apex_hole = ((m > 0.3) & (m < 0.7))
    dilate_px = max(1, int(round(apex_dilate_frac * max(H, W))))
    labels, n_comp = scipy.ndimage.label(apex_hole)
    if n_comp > 0 and dilate_px > 1:
        sizes = scipy.ndimage.sum(apex_hole, labels, index=np.arange(1, n_comp + 1))
        big_ids = np.flatnonzero(sizes >= 5e-4 * H * W) + 1
        if big_ids.size:
            big = np.isin(labels, big_ids)
            big = scipy.ndimage.binary_dilation(big, iterations=dilate_px)
            apex_hole = apex_hole | big
    apex_hole = apex_hole.astype(np.float32)
    apex_w = scipy.ndimage.gaussian_filter(apex_hole, sigma=20.0)
    apex_w = np.clip(apex_w, 0.0, 1.0)
    inside_content = img_np_b * (1.0 - apex_w[None, None]) + img_apex * apex_w[None, None]

    # Outside fade: distance from the binary disk silhouette, fading to black.
    binary_mask = (m > 0.3)
    dist_outside = scipy.ndimage.distance_transform_edt(~binary_mask)
    outside_alpha = np.clip(1.0 - dist_outside / outside_fade_dist, 0.0, 1.0)
    outside_alpha = scipy.ndimage.gaussian_filter(outside_alpha, sigma=8.0)
    outside_alpha = np.clip(outside_alpha, 0.0, 1.0)
    outside_content = img_out * outside_alpha[None, None]

    # Soft blend between inside and outside content along the silhouette.
    soft_m = scipy.ndimage.gaussian_filter(binary_mask.astype(np.float32),
                                           sigma=edge_blend_sigma)
    soft_m = np.clip(soft_m, 0.0, 1.0)

    out = soft_m[None, None] * inside_content + (1.0 - soft_m[None, None]) * outside_content
    out = np.clip(out, 0, 1).astype(np.float32) * (hi - lo) + lo
    return torch.from_numpy(out).to(recovered.device, dtype=out_dtype)


def view_lod(image, warp, leveln=5, padding_mode='border'):
    """
    LOD-aware (mip-map style) forward warp.

    For each output pixel, the local frequency cap is set by the Jacobian
    norm of the UV map: pixels in regions where the warp is locally
    contracting (e.g. the inner edge of the conic mirror's polar unwrap)
    are sampled from a coarser pyramid level so they don't alias. Pixels
    where the warp is non-contracting sample from level 0 — i.e. the
    behavior collapses to plain bilinear ``view_simple`` sampling.

    Implementation: build a Gaussian pyramid of ``image``, upsample each
    level to full resolution and stack along a new pyramid axis to form a
    (B, C, L, H, W) volume, then sample with a single 3D ``grid_sample``
    over (u, v, lod). LOD is computed from ``_compute_lod_level``.

    Args:
        image: Input image (B, C, H, W) or (C, H, W).
        warp:  Warp field (1, 3, H, W) with UV in [0, 1].
        leveln: Number of pyramid levels (default 5).
        padding_mode: Pass-through to grid_sample for out-of-range UV;
            'border' is preferred over 'zeros' so identity-warp regions
            don't get blackened at the edges.

    Returns:
        Warped image with the same leading shape as the input image.
    """
    squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True

    device = image.device
    dtype = image.dtype
    image_f = image.float()

    # Gaussian pyramid: gp[0] = full res, gp[i] = down-sampled by 2^i.
    gp = [image_f]
    for _ in range(leveln - 1):
        gp.append(pyrDown(gp[-1]))

    # Upsample every level back to full res and stack along a pyramid axis.
    layers = pyrStack(gp, dim=2)  # (B, C, L, H, W)
    if layers.ndim == 4:
        layers = layers.unsqueeze(0)

    # Build the (u, v, lod) sampling grid in normalized [-1, 1] coords.
    warp = warp.to(device=device, dtype=torch.float32)
    h, w = warp.shape[-2:]
    L = leveln

    mapping = max(h, w) * warp[:, :2]                  # UV in pixel units
    lod = _compute_lod_max_col(mapping, maxLOD=L - 1)   # 0 where the warp is non-contracting
    lod_norm = 2.0 * lod / max(L - 1, 1) - 1.0          # → [-1, 1]
    lod_norm = lod_norm.unsqueeze(0).unsqueeze(-1)      # (1, H, W, 1)

    uv = warp[:, :2].permute(0, 2, 3, 1)             # (1, H, W, 2)
    uv = 2.0 * uv - 1.0                              # → [-1, 1]

    # 5D grid_sample expects last dim = (x, y, z) where (x, y, z) → (W, H, D).
    # Our volume is (N, C, D=L, H, W), so x = u, y = v, z = lod.
    grid = torch.cat([uv, lod_norm], dim=-1).unsqueeze(1)  # (1, 1, H, W, 3)

    if grid.shape[0] != layers.shape[0]:
        grid = grid.expand(layers.shape[0], -1, -1, -1, -1)

    warped = F.grid_sample(
        layers,
        grid,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(2)  # (B, C, H, W)

    if squeeze:
        warped = warped.squeeze(0)
    return warped.to(dtype)


def view(lp, warp, leveln):
    """
    Forward warp a Laplacian pyramid.
    
    For simple transforms (flips, rotations), we reconstruct the image first
    and then apply 2D warping. This is simpler and more reliable than
    3D LOD-based sampling.
    
    Args:
        lp: Laplacian pyramid (list of tensors)
        warp: Warp field of shape (1, 3, H, W)
        leveln: Number of pyramid levels
    
    Returns:
        Warped image tensor (C, H, W)
    """
    # Reconstruct image from Laplacian pyramid
    gp = Laplacian2Gaussian(lp)
    img = gp[0]  # Full resolution image
    
    # Apply simple 2D warp
    warped = view_simple(img, warp)
    
    return warped


# =============================================================================
# Inverse/Backward Warping
# =============================================================================

def inverse_view(im, warp, leveln):
    """
    Inverse warp using backpropagation to find the pre-warp pyramid.
    
    Args:
        im: Image tensor (C, H, W)
        warp: Warp field of shape (1, 3, H, W)
        leveln: Number of pyramid levels
    
    Returns:
        Laplacian pyramid that, when forward warped, produces im
    """
    c, h, w = im.shape
    device = im.device
    dtype = im.dtype
    
    warp = warp.to(device=device, dtype=torch.float32)
    grid = _get_grid(warp, maxLOD=leveln - 1).float()
    
    # Handle NaN in grid
    valid_mask = ~torch.isnan(grid[..., 0])
    safe_grid = torch.where(torch.isnan(grid), torch.tensor(-2.0, device=device), grid)
    
    with torch.enable_grad():
        # Create an empty pyramid with an extra channel for counting
        opt_var = torch.zeros(1, c + 1, h, w, device=device, dtype=torch.float32)
        opt_var = LaplacianPyramid(opt_var, leveln)
        for lvl in opt_var:
            lvl.requires_grad_()
        
        # Convert to Gaussian pyramid
        opt_gp = Laplacian2Gaussian(opt_var)
        
        # Target includes the image and a mask channel
        target = torch.cat([im.float(), torch.ones(1, h, w, device=device)], dim=0)
        
        # Stack and sample
        layers = pyrStack(opt_gp, dim=2).float()
        if layers.ndim == 4:
            layers = layers.unsqueeze(0)
        
        warped = F.grid_sample(
            layers,
            safe_grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True,
        ).squeeze(2)
        
        # Compute loss only on valid regions
        diff = (warped - target.unsqueeze(0)) ** 2
        loss = 0.5 * (diff * valid_mask.unsqueeze(0).float()).sum()
        
        loss.backward()
        
        # Extract gradients
        result = [-lvl.grad.detach() for lvl in opt_var]
        
        # Normalize by the count channel. Per the paper's pseudocode
        # (resources/suppl_images-pseudocode.py): r[:, :c] / r[:, -1:].
        # Don't clamp the denominator — empty regions (no scatter
        # contributions) must produce NaN/Inf, which marks them for
        # nearest-value imputation in the per-level loop below. Without
        # this, "no data" pixels stay at 0 and Laplacian2Gaussian
        # propagates them as black through the reconstruction.
        processed = []
        for r in result:
            num = r[:, :c]
            den = r[:, -1:]
            with torch.no_grad():
                res = num / den
                # Convert ±Inf (from 0/0 with sign) to NaN so impute fires.
                res = torch.where(torch.isfinite(res), res, torch.full_like(res, float('nan')))
                res = res.squeeze(0)
            processed.append(res)
        result = processed

    # Per-level nearest-value imputation, then convert each level to its
    # Laplacian (subtract pyrUp(pyrDown(.))) — matches paper figure
    # (b)–(g) in resources/suppl_images-lpw_interm_steps.pdf.
    for k in range(leveln - 1):
        mask = ~torch.isnan(result[k])
        if mask.any():
            imputed = impute_with_nearest(result[k], mask)
            result[k] = imputed - pyrUp(pyrDown(imputed))
        else:
            result[k] = torch.zeros_like(result[k])

    # Coarsest Gaussian level also needs imputation so the reconstructed
    # base doesn't carry NaN through Laplacian2Gaussian.
    last = result[leveln - 1]
    mask_last = ~torch.isnan(last)
    if mask_last.any():
        result[leveln - 1] = impute_with_nearest(last, mask_last)
    else:
        result[leveln - 1] = torch.zeros_like(last)

    return result


# =============================================================================
# Pyramid Blending
# =============================================================================

def blend_pyramids(lp1, lp2, alpha=0.25):
    """
    Blend two Laplacian pyramids with detail-preserving averaging.
    
    From the paper (Eq. 12-15):
    - Standard average: avg(x,y) = (x + y) / 2
    - Value-weighted average: vavg(x,y) = (|x|x + |y|y) / (|x| + |y|)
    - Final blend: z = avg + alpha * (vavg - avg)
    
    Missing values (NaN) are handled using torch.nanmean().
    
    Args:
        lp1: First Laplacian pyramid
        lp2: Second Laplacian pyramid
        alpha: Interpolation parameter [0, 1]. 
               0 = standard average (blurrier)
               1 = value-weighted average (preserves details, may over-sharpen)
               Recommended: 0.25 to 0.5
    
    Returns:
        Blended Laplacian pyramid
    """
    blended_lp = []
    
    for l1, l2 in zip(lp1, lp2):
        # Ensure same shape
        if l1.shape != l2.shape:
            raise ValueError(f"Pyramid level shape mismatch: {l1.shape} vs {l2.shape}")
        
        # Standard average using nanmean for partial views (Eq. 13)
        stacked = torch.stack([l1, l2], dim=0)
        avg_result = torch.nanmean(stacked, dim=0)
        
        # Value-weighted average (Eq. 14)
        # Replace NaN with 0 for computation
        l1_safe = torch.where(torch.isnan(l1), torch.zeros_like(l1), l1)
        l2_safe = torch.where(torch.isnan(l2), torch.zeros_like(l2), l2)
        
        abs_l1 = torch.abs(l1_safe)
        abs_l2 = torch.abs(l2_safe)
        
        numerator = abs_l1 * l1_safe + abs_l2 * l2_safe
        denominator = abs_l1 + abs_l2
        
        epsilon = 1e-8
        vavg_result = numerator / torch.clamp(denominator, min=epsilon)
        
        # Handle NaN propagation for vavg
        both_nan = torch.isnan(l1) & torch.isnan(l2)
        vavg_result[both_nan] = float('nan')
        
        # Interpolate between avg and vavg (Eq. 15)
        blended_level = avg_result + alpha * (vavg_result - avg_result)
        blended_lp.append(blended_level)
    
    return blended_lp


def blend_pyramids_masked(lp1, lp2, mask, alpha=0.5, w2=0.5):
    """
    Mask-weighted Laplacian pyramid blending.

    Treats ``lp1`` as the canonical/identity pyramid (defined everywhere)
    and ``lp2`` as a partial-view pyramid that is only valid where
    ``mask > 0``. At each pyramid level, downsamples the mask to that
    level's resolution and blends:

        out_l = m_l * detail_blend(L1_l, L2_l) + (1 - m_l) * L1_l

    where detail_blend interpolates between standard average and a
    value-weighted average per the paper (Eqs. 12–15). Downsampling the
    mask per level gives the paper's partial-view behavior: high-frequency
    content from ``lp2`` stays inside the mask, while lower-frequency
    bands transition more smoothly across the boundary.

    Args:
        lp1: Laplacian pyramid (list of tensors, finest → coarsest).
        lp2: Laplacian pyramid with the same shapes as lp1.
        mask: Soft mask at full resolution, shape (1, 1, H, W) or
            broadcastable to ``lp1[0]``.
        alpha: Detail-preservation strength inside the mask (0 = avg only,
            1 = value-weighted only). Default 0.5.

    Returns:
        Blended Laplacian pyramid (list of tensors).
    """
    # Build the mask's Gaussian pyramid (Burt-Adelson multiband blending):
    # each level is pyrDown-ed and re-smoothed with a small box filter that
    # COMPOUNDS down the pyramid, so the blend transition is a few pixels
    # wide at every level — progressively wider in absolute terms at
    # coarser levels. (Bilinearly resizing the full-res mask instead — the
    # previous behavior — keeps the transition at a fixed absolute width,
    # which is sub-pixel at coarse levels: the multiband blend then
    # degenerates to a hard cut and the seam stays visible.)
    m = mask
    if m.ndim == 3:
        m = m.unsqueeze(0)
    m = m.to(lp1[0].device, dtype=torch.float32).clamp(0.0, 1.0)
    mask_levels = []
    for l1 in lp1:
        target_hw = tuple(l1.shape[-2:])
        if tuple(m.shape[-2:]) != target_hw:
            m = pyrDown(m)
            if tuple(m.shape[-2:]) != target_hw:
                m = F.interpolate(m, size=target_hw, mode='bilinear', align_corners=True)
        m = F.avg_pool2d(m, kernel_size=5, stride=1, padding=2).clamp(0.0, 1.0)
        mask_levels.append(m)

    blended = []
    for i, (l1, l2) in enumerate(zip(lp1, lp2)):
        m_l = mask_levels[i]
        m_s = m_l * m_l * (3.0 - 2.0 * m_l)  # smoothstep

        # Broadcast mask to channel count.
        if l1.ndim == 4 and m_s.shape[1] != l1.shape[1]:
            m_s = m_s.expand(-1, l1.shape[1], -1, -1)
        elif l1.ndim == 3:
            # (1, 1, H, W) → (C, H, W)
            m_s = m_s.expand(-1, l1.shape[0], -1, -1).squeeze(0)

        l1_f = l1.float()
        l2_f = l2.float()

        # ``w2`` weights the partial view (lp2) against the canonical view
        # inside the mask. 0.5 is the symmetric paper blend; raising it
        # compensates transforms whose warp compresses lp2's content (e.g.
        # the conic mirror squeezing a full image into a small circle),
        # which drains its Laplacian magnitudes so a symmetric — and
        # especially a magnitude-weighted — blend would systematically
        # favor lp1.
        w1 = 1.0 - w2
        avg = w1 * l1_f + w2 * l2_f
        a1 = w1 * l1_f.abs()
        a2 = w2 * l2_f.abs()
        vavg = (a1 * l1_f + a2 * l2_f) / torch.clamp(a1 + a2, min=1e-8)
        detail = avg + alpha * (vavg - avg)

        out = m_s * detail + (1.0 - m_s) * l1_f
        blended.append(out.to(l1.dtype))
    return blended


def masked_blend(img1, img2, mask, alpha=0.5):
    """
    Mask-weighted blend treating ``img1`` as the always-defined canonical view
    and ``img2`` as a partial view that is only valid where ``mask > 0``.

    - mask=1: detail-preserving value-weighted blend of img1 and img2.
    - mask=0: pass img1 through unchanged (img2 is undefined here, so it
      must not contribute).
    - intermediate: smooth interpolation between the two.

    For self-overlapping warps (e.g. circular rotations) where img1 ≈ img2
    outside the masked region, this matches the previous 50/50 behavior to
    sub-pixel precision. For partial-view warps (e.g. conic mirror) where
    img2 outside the mask is undefined / spurious, this prevents img2's
    out-of-region content from leaking into img1.

    Args:
        img1: Canonical image (B, C, H, W) - valid everywhere.
        img2: Warped partial-view image (B, C, H, W) - valid only inside mask.
        mask: Soft mask (1, 1, H, W) or (B, 1, H, W), values in [0, 1].
        alpha: Detail-preservation strength inside the mask in [0, 1].

    Returns:
        Blended image (B, C, H, W).
    """
    if mask.shape[0] != img1.shape[0]:
        mask = mask.expand(img1.shape[0], -1, -1, -1)
    if mask.shape[1] != img1.shape[1]:
        mask = mask.expand(-1, img1.shape[1], -1, -1)

    mask = mask.to(img1.device, dtype=img1.dtype)
    smooth_mask = mask * mask * (3.0 - 2.0 * mask)  # smoothstep

    avg_result = (img1 + img2) / 2.0

    abs_img1 = torch.abs(img1)
    abs_img2 = torch.abs(img2)
    numerator = abs_img1 * img1 + abs_img2 * img2
    denominator = abs_img1 + abs_img2
    epsilon = 1e-8
    vavg_result = numerator / torch.clamp(denominator, min=epsilon)

    detail_blend = avg_result + alpha * (vavg_result - avg_result)

    # Inside mask: detail-preserving blend.
    # Outside mask: img1 only — img2 is undefined here.
    result = smooth_mask * detail_blend + (1.0 - smooth_mask) * img1

    return result


# =============================================================================
# High-Level API
# =============================================================================

def laplacian_warp_forward(image, warp, leveln=5):
    """
    Apply forward warping to an image.
    
    For simple transforms like flips, this just applies the warp directly.
    The Laplacian pyramid is used internally for quality.
    
    Args:
        image: Input image (B, C, H, W) or (C, H, W)
        warp: Warp field (1, 3, H, W)
        leveln: Number of pyramid levels (kept for API compatibility)
    
    Returns:
        Warped image (B, C, H, W) or (C, H, W)
    """
    # For simple transforms, just use direct 2D warping
    return view_simple(image, warp)


def laplacian_warp_inverse(image, warp, leveln=5):
    """
    Apply inverse Laplacian pyramid warping to an image.
    
    Args:
        image: Input image (B, C, H, W) or (C, H, W)
        warp: Warp field (1, 3, H, W)
        leveln: Number of pyramid levels
    
    Returns:
        Inverse-warped image (B, C, H, W) or (C, H, W)
    """
    squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True
    
    b, c, h, w = image.shape
    outputs = []
    
    for i in range(b):
        img_i = image[i]
        lp = inverse_view(img_i, warp, leveln)
        gp = Laplacian2Gaussian(lp)
        outputs.append(gp[0])
    
    result = torch.stack(outputs)
    
    if squeeze:
        result = result.squeeze(0)
    
    return result


def laplacian_pyramid_blend(img1, img2, alpha=0.25, leveln=5):
    """
    Blend two images using Laplacian pyramid blending with detail preservation.
    
    Args:
        img1: First image (B, C, H, W) or (C, H, W)
        img2: Second image (same shape as img1)
        alpha: Detail preservation parameter [0, 1]
        leveln: Number of pyramid levels
    
    Returns:
        Blended image
    """
    squeeze = False
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        squeeze = True
    
    # Build pyramids
    lp1 = LaplacianPyramid(img1, leveln)
    lp2 = LaplacianPyramid(img2, leveln)
    
    # Blend
    blended_lp = blend_pyramids(lp1, lp2, alpha=alpha)
    
    # Reconstruct
    gp = Laplacian2Gaussian(blended_lp)
    result = gp[0]
    
    if squeeze:
        result = result.squeeze(0)
    
    return result
