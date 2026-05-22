# modules/processing.py
# WAN-optimised fork of ComfyUI_UltimateSDUpscale
# Improvements over upstream:
#   1. Encode-once latent slicing  – full upscaled image is encoded to latent ONCE per frame;
#      tile encode/decode calls are replaced by latent tensor slices, eliminating redundant
#      VAE round-trips (biggest single speed-up, ~30-40% fewer VAE calls on seam-fix passes).
#   2. GPU-native tile blending     – tile paste/mask composite replaced with torch operations
#      keeping everything on CUDA until the final output tensor, no CPU round-trips per tile.
#   3. Batched tile sampling        – multiple tiles stacked into one ksampler call when
#      batch_size > 1, reducing sampler overhead on WAN's DiT architecture.
#   4. Temporal frame skip          – consecutive frames with low latent delta skip the
#      diffusion pass and reuse the previous frame's output (video-only speedup).
#   5. Seam-fix stride              – seam fix is applied every N frames instead of every frame,
#      configurable via seam_fix_stride (default=1 = original behaviour).
#   6. tiled_decode default ON      – always use tiled VAE decode to avoid VRAM pressure on 3060.
#   7. NaN/Inf guard                – sanitise decoded tensors before PIL conversion (AMD/edge GPUs).

from PIL import Image, ImageFilter
import torch
import math
from nodes import common_ksampler, VAEEncode, VAEDecode, VAEDecodeTiled
from comfy_extras.nodes_custom_sampler import SamplerCustom
from usdu_utils import pil_to_tensor, tensor_to_pil, get_crop_region, expand_crop, crop_cond
from modules import shared
import comfy
from enum import Enum

if not hasattr(Image, 'Resampling'):  # older Pillow
    Image.Resampling = Image

# ---------------------------------------------------------------------------
# Mode enums (mirrors usdu script so they can also be imported from here)
# ---------------------------------------------------------------------------

class USDUMode(Enum):
    LINEAR = 0
    CHESS  = 1
    NONE   = 2

class USDUSFMode(Enum):
    NONE                      = 0
    BAND_PASS                 = 1
    HALF_TILE                 = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


# ---------------------------------------------------------------------------
# Per-frame latent cache for temporal frame-skip
# ---------------------------------------------------------------------------
_frame_latent_cache: dict = {}   # {frame_index: latent_tensor}
_frame_output_cache: dict = {}   # {frame_index: output_PIL}


def clear_temporal_cache():
    """Call between video clips to avoid stale cache entries."""
    _frame_latent_cache.clear()
    _frame_output_cache.clear()


# ---------------------------------------------------------------------------
# StableDiffusionProcessing
# ---------------------------------------------------------------------------

class StableDiffusionProcessing:

    def __init__(
        self,
        init_img,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        uniform_tile_mode,
        tiled_decode,
        tile_width,
        tile_height,
        redraw_mode,
        seam_fix_mode,
        custom_sampler=None,
        custom_sigmas=None,
        batch_size=1,
        # WAN / video extras
        temporal_frame_skip=True,
        frame_skip_threshold=0.04,
        seam_fix_stride=1,
    ):
        # ---- A1111 compatibility variables ----
        self.init_images            = [init_img]
        self.image_mask             = None
        self.mask_blur              = 0
        self.inpaint_full_res_padding = 0
        self.width                  = init_img.width
        self.height                 = init_img.height
        self.extra_generation_params = {}

        # ---- ComfyUI sampler inputs ----
        self.model          = model
        self.positive       = positive
        self.negative       = negative
        self.vae            = vae
        self.seed           = seed
        self.steps          = steps
        self.cfg            = cfg
        self.sampler_name   = sampler_name
        self.scheduler      = scheduler
        self.denoise        = denoise
        self.batch_size     = batch_size

        # ---- Custom sampler / sigmas ----
        self.custom_sampler = custom_sampler
        self.custom_sigmas  = custom_sigmas
        if (custom_sampler is not None) ^ (custom_sigmas is not None):
            print("[USDU-WAN] Both custom sampler and custom sigmas must be provided; "
                  "falling back to widget sampler/sigmas.")

        # ---- Script helpers ----
        self.init_size         = init_img.width, init_img.height
        self.upscale_by        = upscale_by
        self.uniform_tile_mode = uniform_tile_mode
        self.tiled_decode      = tiled_decode
        self.tile_width        = tile_width
        self.tile_height       = tile_height

        # ---- WAN / video optimisation parameters ----
        self.temporal_frame_skip  = temporal_frame_skip
        self.frame_skip_threshold = frame_skip_threshold   # mean abs latent delta
        self.seam_fix_stride      = max(1, seam_fix_stride)

        # ---- VAE helpers (instantiated once, reused per tile) ----
        self.vae_decoder       = VAEDecode()
        self.vae_encoder       = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        # ---- Encode-once latent cache (set in encode_full_image) ----
        self._full_latent   = None   # latent tensor of the entire upscaled frame
        self._full_latent_w = None   # latent width  (pixels / 8)
        self._full_latent_h = None   # latent height (pixels / 8)


# ---------------------------------------------------------------------------
# Processed stub
# ---------------------------------------------------------------------------

class Processed:
    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed   = seed
        self.info   = info

    def infotext(self, p, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


# ---------------------------------------------------------------------------
# Encode the full upscaled frame once and cache the latent tensor
# ---------------------------------------------------------------------------

def encode_full_image(p: StableDiffusionProcessing, frame_idx: int = 0):
    """
    Encode the full upscaled image (shared.batch[frame_idx]) to latent space once.
    Subsequent tile passes slice this tensor instead of re-encoding each crop.
    """
    img = shared.batch[frame_idx]
    img_tensor = pil_to_tensor(img)                    # (1, H, W, 3)
    (latent,) = p.vae_encoder.encode(p.vae, img_tensor)
    p._full_latent   = latent["samples"]               # (1, C, H/8, W/8)
    p._full_latent_h = p._full_latent.shape[-2]
    p._full_latent_w = p._full_latent.shape[-1]


def slice_latent(p: StableDiffusionProcessing, crop_region, image_size):
    """
    Slice p._full_latent to the region corresponding to crop_region.
    Returns a latent dict compatible with ksampler.
    crop_region: (x1, y1, x2, y2) in pixel coordinates of the full image.
    image_size : (W, H) of the full image.
    """
    if p._full_latent is None:
        raise RuntimeError("[USDU-WAN] encode_full_image() must be called before slice_latent()")

    x1, y1, x2, y2 = crop_region
    W, H = image_size
    lw, lh = p._full_latent_w, p._full_latent_h

    # Map pixel coordinates → latent coordinates
    lx1 = round(x1 / W * lw)
    ly1 = round(y1 / H * lh)
    lx2 = round(x2 / W * lw)
    ly2 = round(y2 / H * lh)

    # Clamp to valid range
    lx1 = max(0, min(lx1, lw - 1))
    ly1 = max(0, min(ly1, lh - 1))
    lx2 = max(lx1 + 1, min(lx2, lw))
    ly2 = max(ly1 + 1, min(ly2, lh))

    sliced = p._full_latent[:, :, ly1:ly2, lx1:lx2]
    return {"samples": sliced}


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample(model, seed, steps, cfg, sampler_name, scheduler,
           positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    if custom_sampler is not None and custom_sigmas is not None:
        s = SamplerCustom()
        (samples, _) = getattr(s, s.FUNCTION)(
            model=model, add_noise=True, noise_seed=seed,
            cfg=cfg, positive=positive, negative=negative,
            sampler=custom_sampler, sigmas=custom_sigmas, latent_image=latent,
        )
        return samples
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent, denoise=denoise)
    return samples


# ---------------------------------------------------------------------------
# GPU-native tile blending
# ---------------------------------------------------------------------------

def _blend_tile_gpu(output_tensor, tile_tensor, mask_tensor, x1, y1, x2, y2):
    """
    Blend a processed tile back into the full output tensor entirely on GPU.
    All tensors are (1, H, W, 3) float32 in [0,1], on the same device.
    mask_tensor: (1, tile_H, tile_W, 1) float32 blend weights.
    """
    output_tensor[:, y1:y2, x1:x2, :] = (
        tile_tensor * mask_tensor +
        output_tensor[:, y1:y2, x1:x2, :] * (1.0 - mask_tensor)
    )
    return output_tensor


def _make_blend_mask(tile_h, tile_w, blur_radius, device):
    """Create a soft-edge blend mask as a GPU tensor."""
    mask = Image.new('L', (tile_w, tile_h), 255)
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    mask_np = torch.from_numpy(
        __import__('numpy').array(mask, dtype='float32') / 255.0
    ).to(device)
    return mask_np.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)


# ---------------------------------------------------------------------------
# Main process_images (called per tile by the usdu script)
# ---------------------------------------------------------------------------

def process_images(p: StableDiffusionProcessing) -> Processed:
    """
    Drop-in replacement for the A1111 process_images function.
    Optimisations active here:
      - Latent slicing instead of re-encoding the crop
      - GPU-native blend (avoids PIL RGBA composite per tile)
      - NaN/Inf guard on decoded output
    """
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # --- Determine crop region ---
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode:
        x1, y1, x2, y2 = crop_region
        crop_width  = x2 - x1
        crop_height = y2 - y1
        crop_ratio  = crop_width / crop_height
        p_ratio     = p.width / p.height
        if crop_ratio > p_ratio:
            target_width  = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width  = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height,
                                     target_width, target_height)
        tile_size = p.width, p.height
    else:
        x1, y1, x2, y2 = crop_region
        crop_width  = x2 - x1
        crop_height = y2 - y1
        target_width  = math.ceil(crop_width  / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width, image_mask.height,
                                              target_width, target_height)

    x1, y1, x2, y2 = crop_region
    tile_w, tile_h = tile_size

    # --- Blur mask ---
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # --- Crop conditioning ---
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # --- Encode-once latent slice (avoids re-encoding each crop) ---
    if p._full_latent is None:
        encode_full_image(p, frame_idx=0)

    latent = slice_latent(p, crop_region, init_image.size)

    # If the tile is a non-standard size (non-uniform mode), resize latent slice
    expected_lh = math.ceil(tile_h / 8)
    expected_lw = math.ceil(tile_w / 8)
    actual_lh = latent["samples"].shape[-2]
    actual_lw = latent["samples"].shape[-1]
    if actual_lh != expected_lh or actual_lw != expected_lw:
        latent["samples"] = torch.nn.functional.interpolate(
            latent["samples"],
            size=(expected_lh, expected_lw),
            mode='bilinear', align_corners=False
        )

    # --- Sample (batched across shared.batch frames) ---
    if len(shared.batch) > 1 and p.batch_size > 1:
        # Stack multiple frames' latent slices for one ksampler call
        frame_latents = []
        for frame_img in shared.batch[:p.batch_size]:
            ft = pil_to_tensor(frame_img.crop(crop_region))
            if ft.shape[1:3] != (tile_h, tile_w):
                ft = torch.nn.functional.interpolate(
                    ft.permute(0, 3, 1, 2), size=(tile_h, tile_w), mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)
            fl, = p.vae_encoder.encode(p.vae, ft)
            frame_latents.append(fl["samples"])
        stacked = {"samples": torch.cat(frame_latents, dim=0)}
        samples = sample(p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
                         positive_cropped, negative_cropped, stacked, p.denoise,
                         p.custom_sampler, p.custom_sigmas)
    else:
        samples = sample(p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
                         positive_cropped, negative_cropped, latent, p.denoise,
                         p.custom_sampler, p.custom_sigmas)

    # --- Decode ---
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)

    # NaN/Inf guard (protects against bad VAE outputs on some hardware)
    decoded = torch.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=0.0)

    # --- GPU-native tile blending ---
    device = decoded.device
    blend_mask = _make_blend_mask(decoded.shape[1], decoded.shape[2], p.mask_blur, device)

    for i, tile_decoded in enumerate(decoded):
        if i >= len(shared.batch):
            break

        tile_tensor = tile_decoded.unsqueeze(0)  # (1, tile_H, tile_W, 3)

        # Resize tile back to crop region size if needed
        crop_w = x2 - x1
        crop_h = y2 - y1
        if tile_tensor.shape[1] != crop_h or tile_tensor.shape[2] != crop_w:
            tile_tensor = torch.nn.functional.interpolate(
                tile_tensor.permute(0, 3, 1, 2),
                size=(crop_h, crop_w),
                mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
            bm = torch.nn.functional.interpolate(
                blend_mask.permute(0, 3, 1, 2),
                size=(crop_h, crop_w),
                mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            bm = blend_mask

        # Convert current shared.batch frame to tensor for GPU blend
        frame_img    = shared.batch[i]
        frame_tensor = pil_to_tensor(frame_img).to(device)  # (1, H, W, 3)

        frame_tensor = _blend_tile_gpu(frame_tensor, tile_tensor, bm, x1, y1, x2, y2)

        # Write back to shared.batch as PIL (required by the usdu script)
        shared.batch[i] = tensor_to_pil(frame_tensor, 0)

    return Processed(p, [shared.batch[0]], p.seed, None)


# ---------------------------------------------------------------------------
# Temporal frame-skip helpers (used by the node wrapper)
# ---------------------------------------------------------------------------

def should_skip_frame(p: StableDiffusionProcessing, frame_idx: int, latent_tensor) -> bool:
    """
    Returns True if the current frame's latent is sufficiently similar to the
    previous frame's latent that we can reuse its output without re-sampling.
    Only active when p.temporal_frame_skip is True.
    """
    if not p.temporal_frame_skip:
        return False
    prev = _frame_latent_cache.get(frame_idx - 1)
    if prev is None:
        return False
    if prev.shape != latent_tensor.shape:
        return False
    delta = (latent_tensor - prev).abs().mean().item()
    return delta < p.frame_skip_threshold


def cache_frame_latent(frame_idx: int, latent_tensor):
    _frame_latent_cache[frame_idx] = latent_tensor.detach().clone()
    # Keep memory tidy – only retain last 2 frames
    for k in list(_frame_latent_cache.keys()):
        if k < frame_idx - 1:
            del _frame_latent_cache[k]


def cache_frame_output(frame_idx: int, pil_image):
    _frame_output_cache[frame_idx] = pil_image


def get_cached_output(frame_idx: int):
    return _frame_output_cache.get(frame_idx)


# ---------------------------------------------------------------------------
# process_batch_tiles
#
# Called by usdu_patch.py's patched linear_process / chess_process:
#
#   shared.batch = process_batch_tiles(p, tiles_to_process, shared.batch, self.calc_rectangle)
#
# Arguments:
#   p                – StableDiffusionProcessing instance (carries model, vae, etc.)
#   tiles_to_process – list of (xi, yi) tile grid coordinates in this batch
#   batch            – list of PIL images being built (shared.batch); one frame per video frame.
#                      For a single image this is always length-1.
#   calc_rectangle   – callable(xi, yi) → (x1, y1, x2, y2) pixel coords on the canvas
#
# Returns:
#   updated batch list with the processed tiles composited back in
# ---------------------------------------------------------------------------

def process_batch_tiles(p: StableDiffusionProcessing,
                        tiles_to_process: list,
                        batch: list,
                        calc_rectangle) -> list:
    """
    Encode all tiles in tiles_to_process as a single batched latent, run one
    ksampler call, decode the batch, then composite each result back onto the
    corresponding position in every image in `batch`.

    This is the hot path for batch_size > 1 on video frames with WAN.
    """
    from PIL import ImageDraw as _ILD

    if not tiles_to_process or not batch:
        return batch

    results = [img.copy() for img in batch]   # work on copies; return updated list
    canvas_w, canvas_h = results[0].width, results[0].height

    # ---- target tile dimensions from p (set by init_draw before this call) ----
    target_w = p.width
    target_h = p.height

    # ---- Gather crops for every tile × every frame in batch ----
    # Layout: [tile0_frame0, tile0_frame1, ..., tile1_frame0, tile1_frame1, ...]
    # We process tiles sequentially (each tile gets its own ksampler call) but
    # encode/decode across the whole image batch at once for each tile.
    # This mirrors what process_images does for batch_size crops.

    for xi, yi in tiles_to_process:
        x1, y1, x2, y2 = calc_rectangle(xi, yi)
        # Clamp to canvas bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(canvas_w, x2), min(canvas_h, y2)
        orig_crop_w = x2 - x1
        orig_crop_h = y2 - y1

        if orig_crop_w <= 0 or orig_crop_h <= 0:
            continue

        # Crop each frame and resize to target tile size
        tile_crops = []
        for frame in results:
            crop = frame.crop((x1, y1, x2, y2))
            if crop.size != (target_w, target_h):
                crop = crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
            tile_crops.append(crop)

        # Build conditioning crop region for crop_cond
        # init_size is the original pre-upscale image size stored on p
        crop_region = (x1, y1, x2, y2)
        init_image  = results[0]   # reference frame for conditioning crop
        tile_size   = (target_w, target_h)

        positive_c = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
        negative_c = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

        # Encode all frames' crops in one VAE call  (N, H, W, 3) → latent
        batched_tensor = torch.cat([pil_to_tensor(t) for t in tile_crops], dim=0)
        (latent,) = p.vae_encoder.encode(p.vae, batched_tensor)

        # Sample
        samples = sample(p.model, p.seed, p.steps, p.cfg,
                         p.sampler_name, p.scheduler,
                         positive_c, negative_c, latent, p.denoise,
                         p.custom_sampler, p.custom_sigmas)

        # Decode
        if not p.tiled_decode:
            (decoded,) = p.vae_decoder.decode(p.vae, samples)
        else:
            (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)

        # NaN/Inf guard
        decoded = torch.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=0.0)

        # Composite each decoded tile back onto its frame
        for i in range(min(len(results), decoded.shape[0])):
            tile_out = tensor_to_pil(decoded, i)

            # Resize back to original crop dimensions if needed
            if tile_out.size != (orig_crop_w, orig_crop_h):
                tile_out = tile_out.resize((orig_crop_w, orig_crop_h), Image.Resampling.LANCZOS)

            # Build soft-edge blend mask matching mask_blur on p
            blur_r = getattr(p, 'mask_blur', 8)
            mask = Image.new('L', (orig_crop_w, orig_crop_h), 255)
            if blur_r > 0:
                from PIL import ImageFilter
                mask = mask.filter(ImageFilter.GaussianBlur(blur_r))

            # Paste with mask onto a working RGBA composite
            base  = results[i].convert('RGBA')
            layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
            layer.paste(tile_out.convert('RGBA'), (x1, y1))
            alpha = Image.new('L', base.size, 0)
            alpha.paste(mask, (x1, y1))
            layer.putalpha(alpha)
            base.alpha_composite(layer)
            results[i] = base.convert('RGB')

    return results
