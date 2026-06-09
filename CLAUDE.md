# Project notes

Latent generative anamorphoses (LookingGlass-style, Laplacian pyramid warping)
on top of a vendored `diffusers` (`diffusers/src/`). Branch in progress:
`feat/conic`.

## IN PROGRESS: conic mirror geometry fix (continue here)

This work was scoped locally on a MacBook (2026-06-10) and is meant to be
continued on a cloud server (GPU runs).

### Confirmed problem

The physical setup: an apex-up cone mirror sits on the print, viewer looks
straight down. The visible disk (cone silhouette) reflects the **annulus
OUTSIDE the cone's base circle**. The print region UNDER the cone is occluded
by the mirror and can never contribute. Center↔rim inversion holds over the
annulus: disk center (apex) ↔ far edge of annulus, disk rim ↔ ring at the
cone base.

Two conic modes exist, and only one matches this physics:

- **`conic_global`** — `create_conic_mirror_warp` / `_raytrace_conic_mirror`
  in `diffusers/src/diffusers/pipelines/stable_diffusion_3/lod_new.py:395`.
  **Geometrically CORRECT.** Ray-traces a 35° cone with eye (0,0,-1); ground
  hits land at scene radius 0.70 (cone base) → 2.75 (apex reflection),
  strictly outside the base circle. View 2's center under the cone is the
  unused "apex hole" (blur-filled in the inverse). Keep this geometry.

- **`conic`** — `create_conic_inner_mirror_warp` in `lod_new.py:648`.
  **WRONG region.** Compresses the whole of view 2 into the INNER circle
  (radius_ratio 0.27) of view 1 via `r = R·(1 − ρ/edge(θ))`. A real cone
  sits ON that circle, so the encoded content would be hidden under the
  mirror. It's a stylized radial-inversion illusion, not a realizable conic
  anamorphosis.

- **Teaser animation** — `diffusers/make_teaser.py:312` (`--mode conic`)
  uses the inner-mirror warp: it morphs by magnifying image1's inner circle.
  Same flaw. The warpPolar prototype in `diffusers/design_conic_mirror.ipynb`
  also remaps only inside the masked circle — same flaw.

### Planned fix (not yet implemented)

1. Make the mirror view sample the annulus OUTSIDE the cone base of view 1:
   either reuse the ray-traced `conic_global` mapping, or rewrite the
   inner-mirror warp to sample the annulus `[R_base, R_out]` with the
   inversion-style relation `r' = R_base + (1 − ρ/edge)·(R_out − R_base)`.
2. Rewire `make_teaser.py` conic mode: the morph should pull content inward
   from the annulus into the disk; treat image1's central disk as occluded
   (that's where the simulated cone would be drawn), not as the source of
   the reveal.
3. Re-run generation + teaser on the cloud GPU and re-verify with
   `test_conic_warp.py` (see below).

### Verification assets (repo root — DO NOT OVERWRITE the references)

`test_conic_warp.py` tests forward/inverse conic warps against paper refs:
- `input.png` (source rect-mirror image), `output.png` (expected disk image),
  `uv_conic.png` (expected UV map), `inverse_correct.png` (inverse benchmark)
  are READ-ONLY references.
- It writes `test_*` outputs next to itself and prints mean-abs-diff metrics.

### Useful entry points

- Generation: `diffusers/sd3.5.py --transform conic|conic_global`
  (also `flux1_dev.py`, `flux2_dev.py`). Conic knobs: `--conic-radius`,
  `--conic-view2-weight`, `--conic-view2-refine`.
- Pipeline warp dispatch: `apply_laplacian_warp` in
  `diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py:1593`
  (`conic` vs `conic_global` branches; warp caching; LOD sampling;
  `soften_inverse_conic` applied only on the final inverse).
- Teaser/animation: `diffusers/make_teaser.py`.
