# Latent Generative Anamorphoses 

An implementation of **2D latent generative anamorphoses** using Stable Diffusion 3.5, inspired by the LookingGlass paper [CVPR2025].

This project generates **anamorphic images** — single images that reveal different content when viewed from different perspectives or transformations. For example, an image that looks like Einstein when viewed normally, but reveals Marilyn Monroe when rotated or rearranged.

> **Note:** This implementation includes the basic 2D transformations suggested in the LookingGlass paper, such as **circular rotations** (90°, 135°, 180°), **vertical/horizontal flipping**, and the **jigsaw permutation** from Geng et al. The full 3D anamorphosis features from the original paper are not included.

## Citations

If you use this code, please cite the original papers. We thank the authors for their work:

**LookingGlass (Laplacian Pyramid Warping method):**

```bibtex
@misc{chang2025lookingglassgenerativeanamorphoseslaplacian,
      title={LookingGlass: Generative Anamorphoses via Laplacian Pyramid Warping}, 
      author={Pascal Chang and Sergio Sancho and Jingwei Tang and Markus Gross and Vinicius C. Azevedo},
      year={2025},
      eprint={2504.08902},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.08902}, 
}
```

**Visual Anagrams (Jigsaw transform):**

```bibtex
@inproceedings{geng2024visualanagrams,
      title={Visual Anagrams: Generating Multi-View Optical Illusions with Diffusion Models},
      author={Daniel Geng and Inbum Park and Andrew Owens},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2024},
      eprint={2311.17919},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2311.17919},
}
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/LatentGenerativeAnamorphoses.git
cd LatentGenerativeAnamorphoses

# Install dependencies
pip install -e diffusers/
pip install torch transformers accelerate
```

## Usage

Navigate to the `diffusers/` directory and run:

```bash
cd diffusers

python sd3.5.py \
  --style-prompt "a pop art of" \
  --prompt1 "a cat" \
  --prompt2 "a puppy" \
  --transform jigsaw \
  --output-dir "outputs/einstein_marilyn/" \
  --seed 1
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--style-prompt` | `"a pop art of "` | Style prefix applied to both prompts |
| `--prompt1` | `"albert einstein"` | First subject/view prompt |
| `--prompt2` | `"marilyn monroe"` | Second subject/view prompt (revealed after transform) |
| `--transform` | `"jigsaw"` | Anamorphosis transform type (see below) |
| `--output-dir` | `"."` | Directory to save generated images |
| `--seed` | `1` | Random seed for reproducibility |

### Transform Types

| Transform | Description |
|-----------|-------------|
| `vertical` | Vertical flip — image reveals second prompt when flipped upside-down |
| `horizontal` | Horizontal flip — image reveals second prompt when mirrored left-right |
| `90rot` | 90° circular rotation in center region |
| `135rot` | 135° circular rotation in center region |
| `180rot` | 180° circular rotation in center region |
| `jigsaw` | Jigsaw puzzle permutation (from [Geng et al.](https://arxiv.org/abs/2311.17919)) — rearranging tiles reveals second image |

---

## Examples

### Jigsaw Transform: Cat ↔ Puppy (Pop Art)

```bash
python sd3.5.py \
  --transform jigsaw \
  --style-prompt "a pop art of" \
  --prompt1 "a cat" \
  --prompt2 "a puppy" \
  --output-dir "outputs/puppy_cat_pop_art/" \
  --seed 0
```

| View 1 (Cat) | View 2 (Puppy) |
|:---:|:---:|
| ![Cat](diffusers/outputs/puppy_cat_pop_art/generated_image1.png) | ![Puppy](diffusers/outputs/puppy_cat_pop_art/generated_image2.png) |

---

### Jigsaw Transform: Einstein ↔ Marilyn (Pop Art)

```bash
python sd3.5.py \
  --transform jigsaw \
  --style-prompt "a pop art of" \
  --prompt1 "albert einstein" \
  --prompt2 "marilyn monroe" \
  --output-dir "outputs/einstein_marilyn/" \
  --seed 1
```

| View 1 (Einstein) | View 2 (Marilyn) |
|:---:|:---:|
| ![Einstein](diffusers/outputs/einstein_marilyn/generated_image1.png) | ![Marilyn](diffusers/outputs/einstein_marilyn/generated_image2.png) |

---

### Jigsaw Transform: Fruit Bowl ↔ Gorilla

| View 1 (Fruit) | View 2 (Gorilla) |
|:---:|:---:|
| ![Fruit](diffusers/outputs/fruit_gorilla/generated_image1.png) | ![Gorilla](diffusers/outputs/fruit_gorilla/generated_image2.png) |

---

### Rotation Transform: Einstein ↔ Marilyn (135°)

```bash
python sd3.5.py \
  --transform 135rot \
  --style-prompt "a pop art of" \
  --prompt1 "albert einstein" \
  --prompt2 "marilyn monroe" \
  --output-dir "outputs/einstein_marilyn_rot/" \
  --seed 1
```

---

## How It Works

The method uses **Laplacian Pyramid Warping** to blend two diffusion trajectories:

1. **Dual Prompt Encoding**: Encode both prompts separately
2. **Parallel Denoising**: Denoise two latent paths simultaneously  
3. **Transform & Blend**: Apply geometric transform to one view, blend using Laplacian pyramids
4. **Time Travel**: Re-noise and re-denoise for better coherence
5. **Final Decode**: Decode the blended latent to produce the anamorphic image

The result is a single image where applying the inverse transform reveals the second prompt.

---

## License

This project is for research purposes. Please see the original paper for licensing details.
