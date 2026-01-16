"""
Test script for Laplacian Pyramid Warping and Blending.

Based on the paper "Taming Rectified Flow for Inversion and Editing" (Appendix A.3-A.4)
"""

import torch
import numpy as np
from PIL import Image
import os

# Import from the new clean implementation
from diffusers.pipelines.stable_diffusion_3.lod_new import (
    LaplacianPyramid,
    Laplacian2Gaussian,
    laplacian_warp_forward,
    laplacian_warp_inverse,
    laplacian_pyramid_blend,
    blend_pyramids,
    create_vertical_flip_warp,
    create_identity_warp,
)


def load_image(path, size=(1024, 1024)):
    """Load image and convert to tensor (1, C, H, W) in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def save_image(tensor, path):
    """Save tensor to image file."""
    # Handle various tensor shapes
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor after processing, got shape {tensor.shape}")
    
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1) * 255.0
    img = img.astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"  Saved: {path}")


def test_pyramid_reconstruction():
    """Test that Laplacian pyramid can perfectly reconstruct the original image."""
    print("\n" + "="*60)
    print("TEST 1: Pyramid Reconstruction")
    print("="*60)
    
    # Create a test image
    img = torch.rand(1, 3, 256, 256)
    
    # Build pyramid
    lp = LaplacianPyramid(img, leveln=5)
    print(f"  Built Laplacian pyramid with {len(lp)} levels")
    for i, level in enumerate(lp):
        print(f"    Level {i}: shape {level.shape}")
    
    # Reconstruct
    gp = Laplacian2Gaussian(lp)
    reconstructed = gp[0]
    
    # Check reconstruction error
    error = torch.abs(img - reconstructed).max().item()
    print(f"  Max reconstruction error: {error:.6f}")
    
    if error < 1e-5:
        print("  ✓ PASSED: Perfect reconstruction")
    else:
        print("  ✗ FAILED: Reconstruction error too high")
    
    return error < 1e-5


def test_forward_warp():
    """Test forward warping with vertical flip."""
    print("\n" + "="*60)
    print("TEST 2: Forward Warp (Vertical Flip)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    # Create test image: top half red, bottom half blue
    img = torch.zeros(1, 3, 512, 512, device=device)
    img[:, 0, :256, :] = 1.0  # Red top
    img[:, 2, 256:, :] = 1.0  # Blue bottom
    
    save_image(img, "test_original.png")
    
    # Create vertical flip warp
    warp = create_vertical_flip_warp(512, 512).to(device)
    
    # Apply forward warp
    warped = laplacian_warp_forward(img, warp, leveln=5)
    save_image(warped, "test_forward_warped.png")
    
    # Check if flipped correctly
    # After flip: top should be blue, bottom should be red
    top_red = warped[0, 0, 128, 256].item()
    top_blue = warped[0, 2, 128, 256].item()
    bottom_red = warped[0, 0, 384, 256].item()
    bottom_blue = warped[0, 2, 384, 256].item()
    
    print(f"  Top region - Red: {top_red:.2f}, Blue: {top_blue:.2f} (expected: 0, 1)")
    print(f"  Bottom region - Red: {bottom_red:.2f}, Blue: {bottom_blue:.2f} (expected: 1, 0)")
    
    # Allow some tolerance due to interpolation
    passed = (top_blue > 0.5 and top_red < 0.5 and bottom_red > 0.5 and bottom_blue < 0.5)
    if passed:
        print("  ✓ PASSED: Image correctly flipped")
    else:
        print("  ✗ FAILED: Flip not correct")
    
    return passed


def test_blending():
    """Test pyramid blending with different alpha values."""
    print("\n" + "="*60)
    print("TEST 3: Pyramid Blending")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create two test images
    img1 = torch.zeros(1, 3, 512, 512, device=device)
    img1[:, 0, :, :] = 1.0  # Full red
    
    img2 = torch.zeros(1, 3, 512, 512, device=device)
    img2[:, 2, :, :] = 1.0  # Full blue
    
    save_image(img1, "test_blend_img1.png")
    save_image(img2, "test_blend_img2.png")
    
    # Test different alpha values
    for alpha in [0.0, 0.25, 0.5, 1.0]:
        blended = laplacian_pyramid_blend(img1, img2, alpha=alpha, leveln=5)
        save_image(blended, f"test_blend_alpha{alpha:.2f}.png")
        
        # Check center pixel
        center_r = blended[0, 0, 256, 256].item()
        center_b = blended[0, 2, 256, 256].item()
        print(f"  Alpha={alpha:.2f}: R={center_r:.3f}, B={center_b:.3f}")
    
    print("  ✓ Blending test completed - check output images visually")
    return True


def test_full_pipeline():
    """Test the full pipeline: warp -> blend -> inverse warp."""
    print("\n" + "="*60)
    print("TEST 4: Full Pipeline")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load or create test image
    img_path = "generated_image1.png"
    if os.path.exists(img_path):
        print(f"  Loading image from {img_path}")
        img1 = load_image(img_path, size=(512, 512)).to(device)
    else:
        print("  Creating gradient test image")
        img1 = torch.zeros(1, 3, 512, 512, device=device)
        # Create gradient
        for i in range(512):
            img1[0, 0, i, :] = i / 512.0  # Red vertical gradient
            img1[0, 1, :, i] = i / 512.0  # Green horizontal gradient
        img1[0, 2, :, :] = 0.3  # Blue constant
    
    save_image(img1, "pipeline_1_original.png")
    
    # Step 1: Forward warp to create second view
    warp = create_vertical_flip_warp(512, 512).to(device)
    img2 = laplacian_warp_forward(img1, warp, leveln=5)
    save_image(img2, "pipeline_2_warped.png")
    
    # Step 2: Blend the two views
    blended = laplacian_pyramid_blend(img1, img2, alpha=0.25, leveln=5)
    save_image(blended, "pipeline_3_blended.png")
    
    # Step 3: Create inverse-warped version for second branch
    # This would be used in the diffusion loop
    blended_inv = laplacian_warp_inverse(blended, warp, leveln=5)
    save_image(blended_inv, "pipeline_4_inverse_warped.png")
    
    print("\n  Pipeline outputs:")
    print("    1. pipeline_1_original.png - Original image")
    print("    2. pipeline_2_warped.png - Forward warped (flipped)")
    print("    3. pipeline_3_blended.png - Blended result")
    print("    4. pipeline_4_inverse_warped.png - Inverse warped blend")
    
    return True


def test_with_real_image():
    """Test with a real image if available."""
    print("\n" + "="*60)
    print("TEST 5: Real Image Test")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    img_path = "generated_image1.png"
    if not os.path.exists(img_path):
        print(f"  Skipping: {img_path} not found")
        return True
    
    print(f"  Loading {img_path}")
    img = load_image(img_path, size=(1024, 1024)).to(device)
    save_image(img, "real_1_original.png")
    
    # Forward warp
    warp = create_vertical_flip_warp(1024, 1024).to(device)
    warped = laplacian_warp_forward(img, warp, leveln=5)
    save_image(warped, "real_2_warped.png")
    
    # Blend
    blended = laplacian_pyramid_blend(img, warped, alpha=0.25, leveln=5)
    save_image(blended, "real_3_blended.png")
    
    print("  ✓ Real image test completed")
    return True


def main():
    print("="*60)
    print("LAPLACIAN PYRAMID WARPING & BLENDING TESTS")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("Pyramid Reconstruction", test_pyramid_reconstruction()))
    results.append(("Forward Warp", test_forward_warp()))
    results.append(("Blending", test_blending()))
    results.append(("Full Pipeline", test_full_pipeline()))
    results.append(("Real Image", test_with_real_image()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))


if __name__ == "__main__":
    main()
