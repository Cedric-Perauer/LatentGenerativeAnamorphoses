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

def pyrDown(x):
    """Downsample by factor of 2 using averaging."""
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    
    b, c, h, w = x.shape
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    x = x[:, :, :h_even, :w_even]
    
    # Average 2x2 blocks
    x_reshaped = x.view(b, c, h_even // 2, 2, w_even // 2, 2)
    x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5)
    x_blocks = x_permuted.reshape(b, c, h_even // 2, w_even // 2, 4)
    out = torch.nanmean(x_blocks, dim=-1)
    
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
        grid: (1, 1, H, W, 3) tensor with (lod, u, v) coordinates in [-1, 1]
    """
    h, w = warp.shape[-2:]
    
    # Compute LOD level and normalize to (-1, 1)
    mapping = h * warp[:, :2]  # Scale UV to pixel coordinates
    lod = _compute_lod_level(mapping, maxLOD=maxLOD)
    lod = 2 * lod / maxLOD - 1.0  # Normalize to (-1, 1)
    lod = lod.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
    
    # Normalize UV to (-1, 1)
    grid = warp[:, :2].permute(0, 2, 3, 1)  # (1, H, W, 2)
    
    # Mark undefined regions (where UV is exactly 0) with NaN
    mask = torch.all(grid == 0.0, dim=-1, keepdim=True)
    mask = mask.expand(-1, -1, -1, 2)
    grid = grid.clone()
    grid[mask] = float('nan')
    grid = 2 * grid - 1  # Convert from [0,1] to [-1,1]
    
    # Combine into 3D coordinate grid (lod, u, v)
    grid = torch.cat([lod, grid], dim=-1).unsqueeze(1)  # (1, 1, H, W, 3)
    
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
        
        # Normalize by the count channel
        processed = []
        for r in result:
            num = r[:, :c]
            den = r[:, c:]
            den = torch.clamp(torch.abs(den), min=1e-8)
            res = num / den
            res = res.squeeze(0)  # Remove batch dim
            processed.append(res)
        result = processed
    
    # Extract Laplacian pyramid with imputation
    for k in range(leveln - 1):
        mask = ~torch.isnan(result[k])  # True where valid
        imputed = impute_with_nearest(result[k], mask)
        result[k] = imputed - pyrUp(pyrDown(imputed))
        # Restore NaN where originally missing
        result[k][~mask] = float('nan')
    
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


def masked_blend(img1, img2, mask, alpha=0.5):
    """
    Blend two images everywhere, using the mask to control blend strength.
    
    The mask controls the blend type:
    - mask=1 (inside circle): detail-preserving value-weighted blend
    - mask=0 (outside circle): simple 50/50 average blend
    - intermediate values: smooth transition between the two blend types
    
    This ensures the entire image is a coherent blend of both inputs,
    with stronger detail preservation inside the masked region.
    
    Args:
        img1: First image (B, C, H, W) - the "base" image
        img2: Second image (B, C, H, W) - transformed version (already warped)
        mask: Soft mask (1, 1, H, W) or (B, 1, H, W), values in [0, 1]
        alpha: Blending parameter for value-weighted vs standard average inside mask
    
    Returns:
        Blended image (B, C, H, W)
    """
    # Ensure proper dimensions
    if mask.shape[0] != img1.shape[0]:
        mask = mask.expand(img1.shape[0], -1, -1, -1)
    if mask.shape[1] != img1.shape[1]:
        mask = mask.expand(-1, img1.shape[1], -1, -1)
    
    mask = mask.to(img1.device, dtype=img1.dtype)
    
    # Apply smoothstep to the mask for even smoother transitions
    # smoothstep(x) = 3x^2 - 2x^3, gives S-curve for values in [0, 1]
    smooth_mask = mask * mask * (3.0 - 2.0 * mask)
    
    # Standard average (used outside the circle)
    avg_result = (img1 + img2) / 2.0
    
    # Value-weighted average (used inside the circle for detail preservation)
    abs_img1 = torch.abs(img1)
    abs_img2 = torch.abs(img2)
    
    numerator = abs_img1 * img1 + abs_img2 * img2
    denominator = abs_img1 + abs_img2
    
    epsilon = 1e-8
    vavg_result = numerator / torch.clamp(denominator, min=epsilon)
    
    # Detail-preserving blend for inside the circle
    detail_blend = avg_result + alpha * (vavg_result - avg_result)
    
    # Blend everywhere: 
    # - Inside circle (mask=1): use detail-preserving blend
    # - Outside circle (mask=0): use simple average
    # - Transition zone: smooth interpolation
    result = smooth_mask * detail_blend + (1 - smooth_mask) * avg_result
    
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
