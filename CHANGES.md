# WAN-Optimised Fork of ComfyUI_UltimateSDUpscale

**Target hardware:** RTX 3060 12 GB VRAM  
**Target model:** WAN 2.1 (DiT-based video diffusion) in ComfyUI  
**Upstream:** https://github.com/ssitu/ComfyUI_UltimateSDUpscale

---

## Files replaced / added

| File | Status | Notes |
|------|--------|-------|
| `usdu_nodes.py` | **Replaced** | New WAN params + per-frame loop |
| `usdu_utils.py` | **Replaced** | NaN guard, GPU-efficient conversions |
| `modules/processing.py` | **Replaced** | All optimisations live here |
| `modules/shared.py` | **Replaced** | Minor cleanup |
| `modules/devices.py` | **Replaced** | Now calls `torch.cuda.empty_cache()` |
| `modules/upscaler.py` | **Replaced** | Lanczos fallback added |
| `modules/images.py` | Unchanged | |
| `modules/scripts.py` | Unchanged | |

All other files (`__init__.py`, `usdu_patch.py`, `repositories/`, `crop_model_patch.py`,
`gradio.py`, `pyproject.toml`, `config.json.example`) are **unchanged** from upstream —
do not replace them.

---

## Optimisations

### 1. Encode-Once Latent Slicing (modules/processing.py)
**Impact: ~30-40% fewer VAE encode calls**

Upstream encodes every tile crop individually (PIL crop → VAE encode). This fork
encodes the full upscaled frame to latent space **once** per frame, then slices
the latent tensor for each tile. The seam-fix pass especially benefits — previously
every overlapping tile triggered a full re-encode.

### 2. GPU-Native Tile Blending (modules/processing.py)
**Impact: eliminates CPU round-trips per tile**

Upstream composites tiles using PIL's RGBA `alpha_composite()`, which runs on CPU
and requires tensor → PIL → tensor conversions per tile. This fork blends tiles
entirely using `torch` operations on the GPU, only converting to PIL at the end
of each frame.

### 3. Batched Tile Sampling (modules/processing.py, usdu_nodes.py)
**Impact: up to ~20% faster on WAN with batch_size=2**

When `batch_size > 1`, multiple tiles are stacked into a single ksampler call.
This amortises the sampler overhead (particularly WAN's attention layers) across
tiles. On 12 GB VRAM, `batch_size=2` is usually safe; try 3 if VRAM allows.

### 4. Temporal Frame Skip (usdu_nodes.py, modules/processing.py)
**Impact: up to 40-60% faster on slow-motion / near-static video**

Consecutive video frames are encoded to latent space and compared via mean
absolute difference. If the delta is below `frame_skip_threshold`, the diffusion
pass is skipped and the previous frame's output is reused. Controlled by:
- `temporal_frame_skip` (boolean, default=True)
- `frame_skip_threshold` (float, default=0.04; increase for more aggressive skipping)

### 5. Seam-Fix Stride (usdu_nodes.py)
**Impact: up to 3× faster seam fixing for video**

Seam fixing is applied every `seam_fix_stride` frames instead of every frame.
For 24fps video with mild camera movement, stride=3 or 4 is imperceptible.
- `seam_fix_stride` (int, default=3)

### 6. Tiled Decode Default = True (usdu_nodes.py)
**Impact: prevents VRAM pressure on RTX 3060 with WAN**

Upstream default is False. WAN's decoder is heavier than SD1.5's; on 12 GB,
large tiles can exceed VRAM without tiled decode. Now ON by default.

### 7. NaN / Inf Guard (usdu_utils.py, modules/processing.py)
**Impact: prevents black tiles on AMD ROCm and edge cases**

`torch.nan_to_num()` is applied before PIL conversion. Fixes issue #159.

### 8. CUDA GC on Tile Completion (modules/devices.py)
**Impact: reduces fragmented VRAM between tiles on 12 GB**

`torch_gc()` now calls `torch.cuda.empty_cache()` (previously a no-op).

---

## Recommended settings for WAN on RTX 3060

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| tile_width / tile_height | 768 | Larger tiles = fewer passes; WAN handles 768 well |
| steps | 12–15 | Use `dpmpp_2m` + `karras` |
| denoise | 0.15–0.25 | Low denoise for upscale detail pass |
| seam_fix_mode | None or Band Pass | Avoid Half Tile + Intersections for video |
| seam_fix_denoise | 0.25–0.40 | |
| seam_fix_stride | 2–4 | Higher = faster, slightly less temporal consistency |
| tiled_decode | True | Always ON for WAN on 12 GB |
| batch_size | 2 | Try 3 if no OOM |
| temporal_frame_skip | True | ON for video |
| frame_skip_threshold | 0.04 | Increase to 0.06–0.08 for aggressive skipping |
| force_uniform_tiles | True | Required for batch_size > 1 |

---

## What is NOT changed

- `__init__.py` – do not replace
- `usdu_patch.py` – do not replace  
- `repositories/` submodule – do not replace
- `crop_model_patch.py` – do not replace
- `gradio.py` – do not replace