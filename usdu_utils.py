# usdu_utils.py
# WAN-optimised fork of ComfyUI_UltimateSDUpscale
# Changes vs upstream:
#   - tensor_to_pil: NaN/Inf guard (prevents black tiles on any hardware)
#   - pil_to_tensor: explicit float32 cast + contiguous memory layout for GPU efficiency
#   - crop_cond:     unchanged (correct upstream logic preserved)
#   - get_crop_region / expand_crop: unchanged

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Tensor <-> PIL conversions
# ---------------------------------------------------------------------------

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a (1, H, W, 3) float32 tensor in [0, 1]."""
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def tensor_to_pil(tensor: torch.Tensor, index: int = 0) -> Image.Image:
    """
    Convert a (B, H, W, 3) or (H, W, 3) float32 tensor to a PIL RGB image.
    NaN / Inf values are sanitised before conversion.
    """
    if tensor.ndim == 4:
        t = tensor[index]
    else:
        t = tensor

    # NaN / Inf guard – protects against bad VAE output on AMD ROCm and
    # edge cases with WAN's decoder on large tiles.
    t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)

    arr = (t.cpu().float().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Crop / conditioning helpers (unchanged from upstream)
# ---------------------------------------------------------------------------

def get_crop_region(mask: Image.Image, pad: int = 0):
    """Return the bounding box of the white region in a grayscale mask PIL image."""
    mask_arr = np.array(mask)
    rows = np.any(mask_arr > 0, axis=1)
    cols = np.any(mask_arr > 0, axis=0)
    if not rows.any():
        return 0, 0, mask.width, mask.height
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(mask.width,  x2 + 1 + pad)
    y2 = min(mask.height, y2 + 1 + pad)
    return int(x1), int(y1), int(x2), int(y2)


def expand_crop(region, width: int, height: int, target_width: int, target_height: int):
    """
    Expand a crop region to at least target_width × target_height while keeping it
    within the image bounds. Returns (new_region, (actual_w, actual_h)).
    """
    x1, y1, x2, y2 = region
    actual_width  = x2 - x1
    actual_height = y2 - y1

    if actual_width < target_width:
        diff = target_width - actual_width
        x1 = max(0, x1 - diff // 2)
        x2 = min(width, x1 + target_width)
        x1 = max(0, x2 - target_width)
    if actual_height < target_height:
        diff = target_height - actual_height
        y1 = max(0, y1 - diff // 2)
        y2 = min(height, y1 + target_height)
        y1 = max(0, y2 - target_height)

    return (int(x1), int(y1), int(x2), int(y2)), (x2 - x1, y2 - y1)


def crop_cond(cond, crop_region, init_size, image_size, tile_size):
    """
    Adjust conditioning area annotations for the cropped tile.
    Mirrors the upstream crop_cond logic.
    """
    x1, y1, x2, y2 = crop_region
    orig_w, orig_h = image_size
    tile_w, tile_h = tile_size
    init_w, init_h = init_size

    cropped = []
    for emb, meta in cond:
        new_meta = meta.copy()
        if "area" in meta:
            # area format: (h, w, y, x) in latent space units
            ah, aw, ay, ax = meta["area"]
            # Scale from pixel space to latent and adjust
            scale_x = tile_w / (x2 - x1)
            scale_y = tile_h / (y2 - y1)
            new_ax = max(0, round((ax * 8 - x1) * scale_x / 8))
            new_ay = max(0, round((ay * 8 - y1) * scale_y / 8))
            new_aw = round(aw * scale_x)
            new_ah = round(ah * scale_y)
            new_meta["area"] = (new_ah, new_aw, new_ay, new_ax)
        cropped.append([emb, new_meta])
    return cropped
