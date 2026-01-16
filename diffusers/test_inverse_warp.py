import torch
import numpy as np
from PIL import Image
from diffusers.pipelines.stable_diffusion_3.lod import (
    inverse_view, 
    create_vertical_flip_warp, 
    Laplacian2Gaussian,
    LaplacianPyramid,
    view,
)
import os


def apply_laplacian_warp(image, transform_type="vertical", inverse=False):
    """
    Apply Laplacian pyramid-based warping to an image.
    
    Args:
        image: Tensor of shape (B, C, H, W)
        transform_type: Type of transform ("vertical" for vertical flip)
        inverse: If True, apply inverse warp
    
    Returns:
        Warped image tensor of shape (B, C, H, W)
    """
    b, c, h, w = image.shape
    leveln = 5
    
    # Create warp
    if transform_type == "vertical":
        warp = create_vertical_flip_warp(h, w)
    else:
        # Default to identity if unknown
        from diffusers.pipelines.stable_diffusion_3.lod import create_identity_warp
        warp = create_identity_warp(h, w)
    
    warp = warp.to(image.device, dtype=torch.float32)
    
    outputs = []
    for i in range(b):
        img_i = image[i]  # (C, H, W)
        
        if not inverse:
            # Forward warp
            lp = LaplacianPyramid(img_i.unsqueeze(0), leveln)
            warped = view(lp, warp, leveln)
            outputs.append(warped)
        else:
            # Inverse warp
            lp_opt = inverse_view(img_i, warp, leveln)
            gp = Laplacian2Gaussian(lp_opt)
            unwarped = gp[0]  # (C, H, W)
            outputs.append(unwarped)
    
    output = torch.stack(outputs).to(image.dtype)
    return output


def load_image(path, size=(1024, 1024)):
    if not os.path.exists(path):
        # Create a dummy image with a gradient to verify flip
        img = torch.zeros(1, 3, size[0], size[1])
        # Red gradient horizontal
        img[0, 0, :, :] = torch.linspace(0, 1, size[1]).view(1, -1).expand(size[0], size[1])
        # Green gradient vertical
        img[0, 1, :, :] = torch.linspace(0, 1, size[0]).view(-1, 1).expand(size[0], size[1])
        return img
        
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    return img

def save_image(tensor, path):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1) * 255.0
    img = img.astype(np.uint8)
    Image.fromarray(img).save(path)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a test image
    # Top half red, bottom half blue
    img = torch.zeros(1, 3, 1024, 1024).to(device)
    img[:, 0, :512, :] = 1.0 # Red top
    img[:, 2, 512:, :] = 1.0 # Blue bottom
    
    save_image(img, "test_original.png")
    print("Saved test_original.png (top=red, bottom=blue)")
    
    # Create vertical flip warp
    warp = create_vertical_flip_warp(1024, 1024).to(device)
    
    # Test 1: Forward warp using apply_laplacian_warp
    print("\n=== Test 1: Forward Laplacian Warp ===")
    forward_warped = apply_laplacian_warp(img, transform_type="vertical", inverse=False)
    save_image(forward_warped, "test_forward_warp.png")
    print("Saved test_forward_warp.png (should be flipped: top=blue, bottom=red)")
    
    # Check forward warp colors
    top_color_fw = forward_warped[:, :, 256, 512]
    bottom_color_fw = forward_warped[:, :, 768, 512]
    print(f"Forward warp - Top color (should be blue [0,0,1]): R={top_color_fw[0,0]:.3f}, G={top_color_fw[0,1]:.3f}, B={top_color_fw[0,2]:.3f}")
    print(f"Forward warp - Bottom color (should be red [1,0,0]): R={bottom_color_fw[0,0]:.3f}, G={bottom_color_fw[0,1]:.3f}, B={bottom_color_fw[0,2]:.3f}")
    
    # Test 2: Inverse warp using apply_laplacian_warp
    print("\n=== Test 2: Inverse Laplacian Warp ===")
    print("Running inverse_view (this finds x such that warp(x) = img)...")
    inverse_warped = apply_laplacian_warp(img, transform_type="vertical", inverse=True)
    save_image(inverse_warped, "test_inverse_warp.png")
    print("Saved test_inverse_warp.png (should be flipped: top=blue, bottom=red)")
    
    # Check inverse warp colors
    top_color_inv = inverse_warped[:, :, 256, 512]
    bottom_color_inv = inverse_warped[:, :, 768, 512]
    print(f"Inverse warp - Top color (should be blue [0,0,1]): R={top_color_inv[0,0]:.3f}, G={top_color_inv[0,1]:.3f}, B={top_color_inv[0,2]:.3f}")
    print(f"Inverse warp - Bottom color (should be red [1,0,0]): R={bottom_color_inv[0,0]:.3f}, G={bottom_color_inv[0,1]:.3f}, B={bottom_color_inv[0,2]:.3f}")
    
    # Test 3: Round-trip test (forward then inverse should give original)
    print("\n=== Test 3: Round-trip (forward -> inverse) ===")
    round_trip = apply_laplacian_warp(forward_warped, transform_type="vertical", inverse=True)
    save_image(round_trip, "test_round_trip.png")
    print("Saved test_round_trip.png (should look like original: top=red, bottom=blue)")
    
    # Check round-trip colors
    top_color_rt = round_trip[:, :, 256, 512]
    bottom_color_rt = round_trip[:, :, 768, 512]
    print(f"Round-trip - Top color (should be red [1,0,0]): R={top_color_rt[0,0]:.3f}, G={top_color_rt[0,1]:.3f}, B={top_color_rt[0,2]:.3f}")
    print(f"Round-trip - Bottom color (should be blue [0,0,1]): R={bottom_color_rt[0,0]:.3f}, G={bottom_color_rt[0,1]:.3f}, B={bottom_color_rt[0,2]:.3f}")

if __name__ == "__main__":
    main()
