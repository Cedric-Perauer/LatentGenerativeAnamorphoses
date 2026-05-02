import torch
import torch.nn.functional as F

def conv2d(x, kernel):
    # x of shape (b, c, h, w)
    channels = kernel.shape[0]
    padding = kernel.shape[-1] // 2  # kernel size is odd number
    x = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    y = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=channels)
    return y


def gaussian_kernel(num_channels=3):
    kernel = torch.tensor([[[
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1.],
    ]]])
    kernel *= 1 / 256.
    kernel = kernel.repeat(num_channels, 1, 1, 1)
    return kernel


def pyrDown(x):
    b, c, h, w = x.shape
    kernel = gaussian_kernel(num_channels=c)
    y = conv2d(x, kernel).to(x.device)
    y = y[..., ::2, ::2]
    return y


def pyrUp(x):
    b, c, h, w = x.shape
    kernel = gaussian_kernel(num_channels=c)
    y = torch.zeros(b, c, 2 * h, 2 * w, dtype=x.dtype, device=x.device)
    y[..., ::2, ::2] = x
    y = conv2d(y, kernel) * 4
    return y


def pyrStack(pyr, dim=0, interp_mode="nearest"):
    h, w = pyr[0].shape[-2:]
    resized_pyr = [F.interpolate(l, size=(h, w), mode=interp_mode) for l in pyr]
    return torch.stack(resized_pyr, dim=dim)


def LaplacianPyramid(img, leveln):
    lp = []
    for _ in range(leveln - 1):
        next_img = pyrDown(img)
        lp.append(img - pyrUp(next_img))
        img = next_img
    lp.append(img)
    return lp


def Laplacian2Gaussian(lp):
    img = lp[-1]
    gp = [img]
    for lev_img in lp[-2::-1]:
        img = pyrUp(img) + lev_img
        gp.append(img)
    gp.reverse()
    return gp


def _get_grid(warp, maxLOD):
    """
    Takes in numpy array warp of shape (1, 3, 1024, 1024)
    """
    # compute lod and normalize to (-1, 1) as third coordinate
    mapping = 1024 * warp[:, :2]
    lod = _compute_lod_level(mapping, maxLOD=maxLOD)
    lod = 2 * lod / maxLOD - 1.0  # (h, w)
    lod = lod.unsqueeze(0).unsqueeze(-1)

    # normalize uv to (-1, 1)
    grid = torch.tensor(warp[:, :2]).permute(0, 2, 3, 1)  # take channels R, G
    mask = torch.all(grid == 0.0, dim=-1, keepdim=True).expand(-1, -1, -1, 2)
    grid[mask] = torch.nan  # replace undefined with NaN
    grid = 2 * grid - 1

    # combine into a 3D coordinate grid (lod, u, v)
    grid = torch.cat([lod, grid], -1).unsqueeze(1)
    return grid # (1, 1, dim, dim, 3)


def view(lp, warp, leveln):
    # get 3d coordinate grid
    grid = _get_grid(warp, maxLOD=leveln-1)

    # sample values
    layers = pyrStack(lp, dim=-1)
    new_im = F.grid_sample(
        layers, 
        grid, 
        mode='nearest',
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(2)

    # replace by NaN where image is 0
    new_im[new_im == 0.0] = torch.nan
    new_im = torch.nanmean(new_im, 0).half()
    return new_im


def inverse_view(im, warp, leveln):
    c, h, w = im.shape
    grid = _get_grid(warp, maxLOD=leveln - 1)
    with torch.enable_grad():
        # create an empty pyramid (Laplacian or Gaussian doesn't matter)
        opt_var = torch.zeros(1, c + 1, h, w)
        opt_var = LaplacianPyramid(opt_var, leveln)
        for lvl in opt_var:
            lvl.requires_grad_()

        # convert to Gaussian pyramid
        opt_gp = Laplacian2Gaussian(opt_var)
        target = torch.cat([im, torch.ones_like(im[:1])], 0)
        layers = pyrStack(opt_gp, -1).float()
        warped = F.grid_sample(
            layers,
            grid,
            mode='nearest',
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(2)
        loss = 0.5 * ((warped - target) ** 2).sum()
        loss.backward()
        result = [-l.grad.detach() for l in opt_var]
        result = [(r[:, :c] / r[:, -1:]) for  r in result]  # create NaNs when area is 0

    # extract laplacian pyramid
    for k in range(leveln - 1):
        mask = torch.isnan(result[k])
        imputed = impute_batch_with_nearest(result[k], ~mask)
        result[k] = imputed - pyrUp(pyrDown(imputed))
        result[k][mask] = torch.nan
        
    return result