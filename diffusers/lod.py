import torch
import torch.nn.functional as F
import scipy
#from scipy.ndimage 
import numpy as np

def create_identity_warp(height=1024, width=1024):
    """
    Create an identity warp field that doesn't transform the image.
    Returns: warp field of shape (1, 3, H, W) with normalized UV coordinates [0, 1]
    """
    # Create coordinate grids
    v_coords = torch.linspace(0, 1, height).view(-1, 1).expand(height, width)
    u_coords = torch.linspace(0, 1, width).view(1, -1).expand(height, width)
    
    # Stack UV coordinates and add batch dimension
    warp = torch.stack([u_coords, v_coords], dim=0).unsqueeze(0)  # (1, 2, H, W)
    
    # Add third channel (can be zeros for compatibility)
    third_channel = torch.zeros(1, 1, height, width)
    warp = torch.cat([warp, third_channel], dim=1)  # (1, 3, H, W)
    
    return warp

def create_vertical_flip_warp(height=1024, width=1024):
    """
    Create a vertical flip warp field that flips the image upside down.
    Returns: warp field of shape (1, 3, H, W) with normalized UV coordinates [0, 1]
    """
    # Create coordinate grids - flip the v coordinates
    v_coords = torch.linspace(1, 0, height).view(-1, 1).expand(height, width)  # Flipped: 1->0 instead of 0->1
    u_coords = torch.linspace(0, 1, width).view(1, -1).expand(height, width)   # Normal: 0->1
    
    # Stack UV coordinates and add batch dimension
    warp = torch.stack([u_coords, v_coords], dim=0).unsqueeze(0)  # (1, 2, H, W)
    
    # Add third channel (can be zeros for compatibility)
    third_channel = torch.zeros(1, 1, height, width)
    warp = torch.cat([warp, third_channel], dim=1)  # (1, 3, H, W)
    
    return warp

def _compute_lod_level(uv_map: torch.Tensor, max_lod: int) -> torch.Tensor:
    """
    Args:
        uv_map: (H, W, 2) tensor of [u, v] warping coordinates (normalized to [-1, 1])
        max_lod: maximum LOD level (number of pyramid levels - 1)
    Returns:
        lod: (H, W) tensor of LOD level per pixel (float, not integer indices)
    """
    u = uv_map[..., 0]
    v = uv_map[..., 1]
    # Compute gradients of the UV map (central difference)
    u_x = F.pad(u[2:, :] - u[:-2, :], (0, 0, 1, 1)) / 2
    u_y = F.pad(u[:, 2:] - u[:, :-2], (1, 1, 0, 0)) / 2
    v_x = F.pad(v[2:, :] - v[:-2, :], (0, 0, 1, 1)) / 2
    v_y = F.pad(v[:, 2:] - v[:, :-2], (1, 1, 0, 0)) / 2
    # Norm of the Jacobian
    jac_norm = torch.sqrt(u_x ** 2 + u_y ** 2 + v_x ** 2 + v_y ** 2)
    # Avoid log(0)
    jac_norm = torch.clamp(jac_norm, min=1e-6)
    lod = 0.85 * torch.log2(jac_norm)
    lod = torch.clamp(lod, min=0.0, max=max_lod)
    return lod

def LaplacianPyramid(x, leveln,interpolation_method='bilinear'):
    gp = [x]
    for i in range(leveln - 1):
        x = pyrDown(x)
        gp.append(x)

    lp = []
    for i in range(leveln - 1):
        if interpolation_method == 'nearest':
            GE = F.interpolate(gp[i + 1], scale_factor=2, mode=interpolation_method)
        else:
            GE = F.interpolate(gp[i + 1], scale_factor=2, mode=interpolation_method, align_corners=False)
        L = gp[i] - GE
        lp.append(gp[i])  # Store the Laplacian (difference), not the Gaussian
    lp.append(gp[-1])  # Last level is the smallest Gaussian
    return lp


def pyrDown(x):
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
        
    b, c, h, w = x.shape
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    x = x[:, :, :h_even, :w_even]
    
    x_reshaped = x.view(b, c, h_even // 2, 2, w_even // 2, 2)
    x_permuted = x_reshaped.permute(0, 1, 2, 4, 3, 5)
    x_blocks = x_permuted.reshape(b, c, h_even // 2, w_even // 2, 4)
    out = torch.nanmean(x_blocks, dim=-1)
    
    if squeeze:
        out = out.squeeze(0)
    return out

def pyrUp(x,interpolation_method='bilinear'):
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True

    if interpolation_method == 'nearest':
        out = F.interpolate(x, scale_factor=2, mode=interpolation_method)
    else:
        out = F.interpolate(x, scale_factor=2, mode=interpolation_method, align_corners=False)
    
    if squeeze:
        out = out.squeeze(0)
    return out

def Laplacian2Gaussian(lp,interpolation_method='bilinear'):
    gp = [lp[-1]]
    for i in range(len(lp) - 1)[::-1]:
        x = gp[-1]
        squeeze = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True

        if interpolation_method == 'nearest':
            GE = F.interpolate(x, scale_factor=2, mode=interpolation_method)
        else:
            GE = F.interpolate(x, scale_factor=2, mode=interpolation_method, align_corners=False)
        
        if squeeze:
            GE = GE.squeeze(0)

        G = lp[i] + GE
        gp.append(G)
    gp.reverse()
    return gp

def pyrStack(pyramid, dim=-1,interpolation_method='bilinear'):
    """
    Upsamples all levels to the highest resolution and stacks along 'dim'.
    Assumes 'pyramid' is a list of tensors from lowest to highest LOD.
    """
    # Determine target size (highest resolution)
    target_size = pyramid[-1].shape[-2:]  # (H, W)
    upsampled = []
    for level in pyramid:
        if interpolation_method == 'nearest':
            upsampled.append(F.interpolate(level, size=target_size, mode=interpolation_method))
        else:
            upsampled.append(F.interpolate(level, size=target_size, mode=interpolation_method, align_corners=True))
    
    return torch.stack(upsampled, dim=dim)
 

def _get_grid(warp, maxLOD):
    """
    Takes in numpy array warp of shape (1, 3, 1024, 1024)
    """
    # compute lod and normalize to (-1, 1)
    mapping = 1024 * warp[0, :2].permute(1, 2, 0)
    lod = _compute_lod_level(mapping, max_lod=maxLOD)
    lod = 2 * lod / maxLOD - 1.0   # (h, w)
    lod = lod.unsqueeze(0).unsqueeze(-1)

    # normalize uv to (-1, 1)
    grid = warp[:, :2].permute(0, 2, 3, 1)
    mask = torch.all(grid == 0.0, dim=-1, keepdim=True)
    mask = mask.expand(-1, -1, -1, 2)
    grid = grid.clone()
    grid[mask] = float('nan')   # replace undefined with NaN
    grid = 2 * grid - 1

    # combine into a 3D coordinate grid (lod, u, v)
    grid = torch.cat([lod, grid], -1).unsqueeze(1)
    return grid   # (1, 1, dim, dim, 3)


def impute_with_nearest(img, mask=None):
    """
    Fills missing values in 'img' (where mask is True or img is NaN) with nearest values.
    img: tensor of shape [C, H, W] or [H, W]
    mask: boolean tensor (True for missing). Optional.
    """
    # Convert to numpy for convenient processing
    img_np = img.cpu().numpy()
    
    if mask is None:
        mask_np = np.isnan(img_np)
    else:
        mask_np = mask.cpu().numpy() | np.isnan(img_np)

    # Find indices of valid (non-missing) pixels
    if img_np.ndim == 3:
        for c in range(img_np.shape[0]):
            if np.all(mask_np[c]):
                continue
            # Distance transform returns indices of nearest valid pixel for each missing pixel
            _, indices = scipy.ndimage.distance_transform_edt(mask_np[c], return_indices=True)
            filled = img_np[c][indices[0], indices[1]]
            img_np[c][mask_np[c]] = filled[mask_np[c]]
    else:
        if not np.all(mask_np):
            _, indices = scipy.ndimage.distance_transform_edt(mask_np, return_indices=True)
            filled = img_np[indices[0], indices[1]]
            img_np[mask_np] = filled[mask_np]
    # Back to torch
    return torch.from_numpy(img_np).to(img.device)

def view(lp, warp, leveln):
    '''
    warp : the warp field for the transformation (1, 3, H, W) with UV in [0,1]
    lp : the laplacian pyramid to be transformed
    leveln : number of levels in the laplacian pyramid
    
    For simple transforms like flips, we apply the warp directly to each pyramid level
    and reconstruct. This avoids the complexity of 3D LOD-based sampling.
    '''
    # Reconstruct the image from the Laplacian pyramid first
    gp = Laplacian2Gaussian(lp)
    img = gp[0]  # Full resolution image (B, C, H, W) or (C, H, W)
    
    squeeze = False
    if img.ndim == 3:
        img = img.unsqueeze(0)
        squeeze = True
    
    b, c, h, w = img.shape
    
    # Convert warp from [0,1] to [-1,1] for grid_sample
    # warp is (1, 3, H, W), we need (B, H, W, 2)
    grid = warp[:, :2].permute(0, 2, 3, 1)  # (1, H, W, 2) - (u, v)
    grid = 2 * grid - 1  # Convert from [0,1] to [-1,1]
    
    # grid_sample expects (x, y) where x is horizontal (width) and y is vertical (height)
    # Our warp has (u, v) where u is horizontal and v is vertical, so order is correct
    
    # Expand grid to batch size if needed
    if grid.shape[0] != b:
        grid = grid.expand(b, -1, -1, -1)
    
    # Ensure dtype match
    grid = grid.to(img.dtype)
    
    # Apply the warp using grid_sample
    warped = F.grid_sample(
        img,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    
    if squeeze:
        warped = warped.squeeze(0)
    
    return warped


def inverse_view(im, warp, leveln):
    c, h, w = im.shape
    grid = _get_grid(warp, maxLOD=leveln - 1).float()

    # Handle NaNs in grid for optimization stability
    valid_mask = ~torch.isnan(grid[..., 0]) # (1, 1, H, W)
    safe_grid = torch.where(torch.isnan(grid), torch.tensor(-2.0, device=grid.device, dtype=grid.dtype), grid)

    with torch.enable_grad():
        # create an empty pyramid
        opt_var = torch.zeros(1, c + 1, h, w, device=im.device)
        opt_var = LaplacianPyramid(opt_var, leveln)
        for lvl in opt_var:
            lvl.requires_grad_()

        # convert to Gaussian pyramid
        opt_gp = Laplacian2Gaussian(opt_var)
        target = torch.cat([im, torch.ones_like(im[:1])])
        layers = pyrStack(opt_gp, 2).float()
        warped = F.grid_sample(
            layers,
            safe_grid,
            mode='bilinear',
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(2)
        
        diff = (warped - target) ** 2
        loss = 0.5 * (diff * valid_mask).sum()
        
        loss.backward()
        result = [l.grad.detach() for l in opt_var]
        
        # Safe division
        processed_result = []
        for r in result:
            num = r[0, :c]
            den = r[0, -1:]
            res = num / den
            # Replace Inf with NaN so they are imputed
            res[torch.isinf(res)] = float('nan')
            processed_result.append(res)
        result = processed_result

    # extract laplacian pyramid
    for k in range(leveln - 1):
        imputed = impute_with_nearest(result[k])
        result[k] = imputed - pyrUp(pyrDown(imputed))

    return result
