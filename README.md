# Latent Generative Anamorphoses 

An implementation of **2D latent generative anamorphoses** using Stable Diffusion 3.5, inspired by the [LookingGlass](https://arxiv.org/abs/2504.08902) paper [CVPR2025].

This project generates **anamorphic images** — single images that reveal different content when viewed from different perspectives or transformations. For example, an image that looks like Einstein when viewed normally, but reveals Marilyn Monroe when rotated or rearranged.

> **Note:** This implementation includes the basic 2D transformations suggested in the LookingGlass paper, such as **circular rotations** (90°, 135°, 180°), **vertical/horizontal flipping**, and the **jigsaw permutation** from Geng et al. The full 3D anamorphosis features from the original paper are not included.


| Jigsaw Puzzle : View 1 (Cat) | Jigsaw Puzzle : View 2 (Puppy) |
|:---:|:---:|
| <img src="diffusers/outputs/puppy_cat_pop_art/generated_image1.png" width="512"/> | <img src="diffusers/outputs/puppy_cat_pop_art/generated_image2.png" width="512"/> |


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


### 90 degree rotation transform: Einstein ↔ Marilyn (Pop Art)

- Style-Prompt : "a pop art of" 
- Prompt1 : "albert einstein"
- Prompt2 : "marilyn monroe"


| View 1 (Einstein) | View 2 (Marilyn) |
|:---:|:---:|
| <img src="diffusers/outputs/einstein_marilyn/generated_image1.png" width="512"/> | <img src="diffusers/outputs/einstein_marilyn/generated_image2.png" width="512"/> |

---

### Jigsaw Transform: Fruit Bowl ↔ Gorilla

- Style-Prompt : "an oil painting of" 
- Prompt1 : "a bowl of fruits"
- Prompt2 : "a gorilla"

| View 1 (Fruit) | View 2 (Gorilla) |
|:---:|:---:|
| <img src="diffusers/outputs/fruit_gorilla/generated_image1.png" width="512"/> | <img src="diffusers/outputs/fruit_gorilla/generated_image2.png" width="512"/> |

---

### 90 degree rotation transform: Village ↔ Horse 

- Style-Prompt : "a painting of" 
- Prompt1 : "a village"
- Prompt2 : "a horse"

| View 1 (Village) | View 2 (Horse) |
|:---:|:---:|
| <img src="diffusers/outputs/horse_village/generated_image1.png" width="512"/> | <img src="diffusers/outputs/horse_village/generated_image2.png" width="512"/> |


### Jigsaw puzzle transform: Puppy ↔ Cat 

- Style-Prompt : "a water color painting of" 
- Prompt1 : "a puppy"
- Prompt2 : "a cat"

| View 1 (Puppy) | View 2 (Cat) |
|:---:|:---:|
| <img src="diffusers/outputs/jigsaw/puppy.png" width="512"/> | <img src="diffusers/outputs/jigsaw/cat.png" width="512"/> |



### 90 degree rotation transform: Village ↔ Ship 

- Style-Prompt : "an oil painting of" 
- Prompt1 : "a ship"
- Prompt2 : "a village in the mountains"

| View 1 (Ship) | View 2 (Village) |
|:---:|:---:|
| <img src="diffusers/outputs/ship_town/generated_image1_3.png" width="512"/> | <img src="diffusers/outputs/ship_town/generated_image2_3.png" width="512"/> |

### Jigsaw puzzle transform: Flowers ↔ Bird 

- Style-Prompt : "a water color painting of" 
- Prompt1 : "flowers"
- Prompt2 : "a bird"

| View 1 (Flowers) | View 2 (Bird) |
|:---:|:---:|
| <img src="diffusers/outputs/jigsaw/generated_image1_flowers.png" width="512"/> | <img src="diffusers/outputs/jigsaw/generated_image2_flowers.png" width="512"/> |



### 90 degree inner circular transform: Cave ↔ Parrot 

- Style-Prompt : "a rendering of" 
- Prompt1 : "an icy cave"
- Prompt2 : "a parrot"

| View 1 (Cave) | View 2 (Parrot) |
|:---:|:---:|
| <img src="diffusers/outputs/parrot_cave/generated_image1.png" width="512"/> | <img src="diffusers/outputs/parrot_cave/generated_image2.png" width="512"/> |


### 90 degree rotation transform: Man ↔ Camp Fire 

- Style-Prompt : "an oil painting of" 
- Prompt1 : "people at a camp fire"
- Prompt2 : "a man"

| View 1 (Man) | View 2 (Camp Fire) |
|:---:|:---:|
| <img src="diffusers/outputs/man_campfire/generated_image1.png" width="512"/> | <img src="diffusers/outputs/man_campfire/generated_image2.png" width="512"/> |







---

## How It Works

The method uses **Laplacian Pyramid Warping** to blend two diffusion trajectories:

1. **Dual Prompt Encoding**: Encode both prompts separately
2. **Parallel Denoising**: Denoise two latent paths simultaneously  
3. **Transform & Blend**: Apply geometric transform to one view, blend using Laplacian pyramids
4. **Time Travel**: Re-noise and re-denoise for better coherence
5. **Final Decode**: Decode the blended latent to produce the anamorphic image

The result is a single image where applying the inverse transform reveals the second prompt.

For more details please see the explanation [on my website (to be added)]() or the [original paper](https://arxiv.org/abs/2504.08902)

---

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


