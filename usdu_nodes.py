# usdu_nodes.py
# WAN-optimised fork of ComfyUI_UltimateSDUpscale
# Adds: temporal_frame_skip, frame_skip_threshold, seam_fix_stride
# Changes: tiled_decode default → True, seam_fix_mode default → "None",
#          seam_fix_denoise default → 0.35 (lighter pass for video)

import logging
from contextlib import contextmanager

import torch
import comfy

from usdu_patch import usdu
from usdu_utils import tensor_to_pil, pil_to_tensor
from modules.processing import (
    StableDiffusionProcessing,
    clear_temporal_cache,
    encode_full_image,
    cache_frame_latent,
    cache_frame_output,
    get_cached_output,
)
import modules.shared as shared
from modules.upscaler import UpscalerData
from nodes import VAEEncode

logger = logging.getLogger(__name__)


@contextmanager
def suppress_logging(level=logging.CRITICAL + 1):
    root_logger = logging.getLogger()
    old_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(level)
    try:
        yield
    finally:
        root_logger.setLevel(old_level)


MAX_RESOLUTION = 8192

MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess":  usdu.USDUMode.CHESS,
    "None":   usdu.USDUMode.NONE,
}

SEAM_FIX_MODES = {
    "None":                      usdu.USDUSFMode.NONE,
    "Band Pass":                 usdu.USDUSFMode.BAND_PASS,
    "Half Tile":                 usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


def USDU_base_inputs():
    required = [
        ("image", ("IMAGE", {"tooltip": "The image (or video frame batch) to upscale."})),
        # Sampling
        ("model",       ("MODEL",        {"tooltip": "The model to use for image-to-image."})),
        ("positive",    ("CONDITIONING", {"tooltip": "Positive conditioning for each tile."})),
        ("negative",    ("CONDITIONING", {"tooltip": "Negative conditioning for each tile."})),
        ("vae",         ("VAE",          {"tooltip": "VAE model to use for tiles."})),
        ("upscale_by",  ("FLOAT",  {"default": 2,    "min": 0.05, "max": 4,    "step": 0.05})),
        ("seed",        ("INT",    {"default": 0,    "min": 0,    "max": 0xffffffffffffffff})),
        ("steps",       ("INT",    {"default": 15,   "min": 1,    "max": 10000, "step": 1,
                                    "tooltip": "Fewer steps (12-16) work well for WAN with dpmpp_2m/karras."})),
        ("cfg",         ("FLOAT",  {"default": 6.0,  "min": 0.0,  "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS, {})),
        ("scheduler",   (comfy.samplers.KSampler.SCHEDULERS, {})),
        ("denoise",     ("FLOAT",  {"default": 0.2,  "min": 0.0,  "max": 1.0,  "step": 0.01})),
        # Upscale
        ("upscale_model", ("UPSCALE_MODEL", {})),
        ("mode_type",   (list(MODES.keys()), {})),
        ("tile_width",  ("INT",    {"default": 768,  "min": 64, "max": MAX_RESOLUTION, "step": 8,
                                    "tooltip": "768 or 1024 recommended for WAN (larger tiles = fewer passes)."})),
        ("tile_height", ("INT",    {"default": 768,  "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur",   ("INT",    {"default": 8,    "min": 0,  "max": 64,   "step": 1})),
        ("tile_padding",("INT",    {"default": 32,   "min": 0,  "max": MAX_RESOLUTION, "step": 8})),
        # Seam fix – lighter defaults for video
        ("seam_fix_mode",    (list(SEAM_FIX_MODES.keys()), {})),
        ("seam_fix_denoise", ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Lower values (0.2-0.4) are much faster for video."})),
        ("seam_fix_width",   ("INT",   {"default": 64,  "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur",("INT",  {"default": 8,   "min": 0, "max": 64,  "step": 1})),
        ("seam_fix_padding", ("INT",   {"default": 16,  "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True})),
        # tiled_decode default is now TRUE – prevents VRAM pressure on RTX 3060 with WAN
        ("tiled_decode",        ("BOOLEAN", {"default": True,
                                             "tooltip": "Keep ON for WAN on 12 GB VRAM."})),
        ("batch_size",          ("INT",     {"default": 1,  "min": 1, "max": 4096, "step": 1,
                                             "tooltip": "Tiles per sampler call. Try 2 on 12 GB with WAN."})),
        # ---- WAN / video optimisation extras ----
        ("temporal_frame_skip",  ("BOOLEAN", {"default": True,
                                              "tooltip": "Skip diffusion on nearly-identical consecutive frames (video speedup)."})),
        ("frame_skip_threshold", ("FLOAT",   {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.005,
                                              "tooltip": "Mean latent delta below which a frame is skipped. 0.02=aggressive, 0.06=conservative."})),
        ("seam_fix_stride",      ("INT",     {"default": 3,   "min": 1,   "max": 60,  "step": 1,
                                              "tooltip": "Apply seam fix every N frames. 1=every frame, 3=every 3rd frame."})),
    ]
    optional = []
    return required, optional


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type_ in required:
            inputs["required"][name] = type_
    if optional:
        inputs["optional"] = {}
        for name, type_ in optional:
            inputs["optional"][name] = type_
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            return


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            return


# ---------------------------------------------------------------------------
# Temporal frame skip helper (standalone, no dependency on processing module)
# ---------------------------------------------------------------------------

_prev_latent_cache: dict = {}


def _check_temporal_skip(vae, frame_pil, frame_idx: int,
                          threshold: float) -> bool:
    """
    Encode frame to latent, compare with previous frame's latent.
    Returns True if the frame is similar enough to skip diffusion.
    Caches the latent for next frame's comparison.
    """
    enc = VAEEncode()
    ft = pil_to_tensor(frame_pil)
    (fl,) = enc.encode(vae, ft)
    current = fl["samples"]

    skip = False
    if frame_idx > 0 and frame_idx in _prev_latent_cache:
        prev = _prev_latent_cache[frame_idx - 1]
        if prev.shape == current.shape:
            delta = (current - prev).abs().mean().item()
            skip = (delta < threshold)

    # Store for next frame
    _prev_latent_cache[frame_idx] = current.detach().clone()
    # Prune old entries
    for k in list(_prev_latent_cache.keys()):
        if k < frame_idx - 1:
            del _prev_latent_cache[k]

    return skip


# ---------------------------------------------------------------------------
# Core upscale implementation
# ---------------------------------------------------------------------------

class UltimateSDUpscale:

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image / video frame batch.",)
    DESCRIPTION = (
        "WAN-optimised Ultimate SD Upscale. "
        "Adds temporal frame-skip, seam-fix stride, encode-once latent slicing, "
        "GPU-native blending, and tiled decode ON by default."
    )

    def upscale(
        self,
        image, model, positive, negative, vae,
        upscale_by, seed, steps, cfg, sampler_name, scheduler, denoise,
        upscale_model,
        mode_type, tile_width, tile_height, mask_blur, tile_padding,
        seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur, seam_fix_width, seam_fix_padding,
        force_uniform_tiles, tiled_decode, batch_size=1,
        temporal_frame_skip=True, frame_skip_threshold=0.04, seam_fix_stride=3,
        custom_sampler=None, custom_sigmas=None,
    ):
        redraw_mode      = MODES[mode_type]
        seam_fix_mode_v  = SEAM_FIX_MODES[seam_fix_mode]

        assert batch_size == 1 or force_uniform_tiles, (
            "batch_size > 1 requires force_uniform_tiles=True."
        )

        # Upscaler stubs
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model

        # Build per-frame PIL list
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        # Clear temporal cache at the start of each node execution
        clear_temporal_cache()
        _prev_latent_cache.clear()

        logger.debug(
            "UltimateSDUpscale.upscale() frames=%d batch_size=%d "
            "temporal_skip=%s threshold=%.3f seam_stride=%d",
            len(shared.batch), batch_size,
            temporal_frame_skip, frame_skip_threshold, seam_fix_stride,
        )

        output_frames = []
        all_frames = list(shared.batch)  # snapshot

        for frame_idx, frame_pil in enumerate(all_frames):

            # ---- Temporal frame skip check ----
            if temporal_frame_skip and frame_idx > 0:
                skip = _check_temporal_skip(
                    vae, frame_pil, frame_idx, frame_skip_threshold
                )
                if skip:
                    prev_out = get_cached_output(frame_idx - 1)
                    if prev_out is not None:
                        logger.debug(
                            "Frame %d: latent delta below threshold (%.3f), reusing prev output.",
                            frame_idx, frame_skip_threshold,
                        )
                        output_frames.append(prev_out)
                        cache_frame_output(frame_idx, prev_out)
                        continue

            # ---- Set shared.batch to just this frame for the tile loop ----
            shared.batch = [frame_pil]

            # ---- Build processing object for this frame ----
            sdprocessing = StableDiffusionProcessing(
                frame_pil, model, positive, negative, vae,
                seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by,
                force_uniform_tiles, tiled_decode,
                tile_width, tile_height, redraw_mode, seam_fix_mode_v,
                custom_sampler, custom_sigmas, batch_size,
                temporal_frame_skip=temporal_frame_skip,
                frame_skip_threshold=frame_skip_threshold,
                seam_fix_stride=seam_fix_stride,
            )

            # Encode full frame once before tile loop (encode-once optimisation)
            encode_full_image(sdprocessing, frame_idx=0)

            # Determine if seam fix should run this frame
            apply_seam_fix = (frame_idx % seam_fix_stride == 0)
            actual_seam_fix = seam_fix_mode_v if apply_seam_fix else usdu.USDUSFMode.NONE

            with suppress_logging():
                script = usdu.Script()
                script.run(
                    p=sdprocessing, _=None,
                    tile_width=tile_width, tile_height=tile_height,
                    mask_blur=mask_blur, padding=tile_padding,
                    seams_fix_width=seam_fix_width,
                    seams_fix_denoise=seam_fix_denoise,
                    seams_fix_padding=seam_fix_padding,
                    upscaler_index=0, save_upscaled_image=False,
                    redraw_mode=redraw_mode,
                    save_seams_fix_image=False,
                    seams_fix_mask_blur=seam_fix_mask_blur,
                    seams_fix_type=actual_seam_fix,
                    target_size_type=2,
                    custom_width=None, custom_height=None,
                    custom_scale=upscale_by,
                )

            result_pil = shared.batch[0]
            output_frames.append(result_pil)
            cache_frame_output(frame_idx, result_pil)

        # Reassemble output tensor
        tensors = [pil_to_tensor(f) for f in output_frames]
        return (torch.cat(tensors, dim=0),)


# ---------------------------------------------------------------------------
# No-Upscale variant
# ---------------------------------------------------------------------------

class UltimateSDUpscaleNoUpscale(UltimateSDUpscale):

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final refined image / video frame batch.",)
    DESCRIPTION = "WAN-optimised USDU (No Upscale). Runs tiled i2i without prior upscaling."

    def upscale(
        self, upscaled_image, model, positive, negative, vae, seed,
        steps, cfg, sampler_name, scheduler, denoise,
        mode_type, tile_width, tile_height, mask_blur, tile_padding,
        seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur, seam_fix_width, seam_fix_padding,
        force_uniform_tiles, tiled_decode, batch_size=1,
        temporal_frame_skip=True, frame_skip_threshold=0.04, seam_fix_stride=3,
    ):
        return super().upscale(
            upscaled_image, model, positive, negative, vae,
            1.0, seed, steps, cfg, sampler_name, scheduler, denoise,
            None,
            mode_type, tile_width, tile_height, mask_blur, tile_padding,
            seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur, seam_fix_width, seam_fix_padding,
            force_uniform_tiles, tiled_decode, batch_size,
            temporal_frame_skip, frame_skip_threshold, seam_fix_stride,
        )


# ---------------------------------------------------------------------------
# Custom-Sample variant
# ---------------------------------------------------------------------------

class UltimateSDUpscaleCustomSample(UltimateSDUpscale):

    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        optional.append(("upscale_model",  ("UPSCALE_MODEL", {"tooltip": "Optional. Omit to use Lanczos."})))
        optional.append(("custom_sampler", ("SAMPLER",       {"tooltip": "Custom sampler; requires custom_sigmas."})))
        optional.append(("custom_sigmas",  ("SIGMAS",        {"tooltip": "Custom sigmas; requires custom_sampler."})))
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image / video frame batch.",)
    DESCRIPTION = "WAN-optimised USDU (Custom Sample)."

    def upscale(
        self, image, model, positive, negative, vae,
        upscale_by, seed, steps, cfg, sampler_name, scheduler, denoise,
        mode_type, tile_width, tile_height, mask_blur, tile_padding,
        seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur, seam_fix_width, seam_fix_padding,
        force_uniform_tiles, tiled_decode, batch_size=1,
        temporal_frame_skip=True, frame_skip_threshold=0.04, seam_fix_stride=3,
        upscale_model=None, custom_sampler=None, custom_sigmas=None,
    ):
        return super().upscale(
            image, model, positive, negative, vae,
            upscale_by, seed, steps, cfg, sampler_name, scheduler, denoise,
            upscale_model,
            mode_type, tile_width, tile_height, mask_blur, tile_padding,
            seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur, seam_fix_width, seam_fix_padding,
            force_uniform_tiles, tiled_decode, batch_size,
            temporal_frame_skip, frame_skip_threshold, seam_fix_stride,
            custom_sampler, custom_sigmas,
        )


# ---------------------------------------------------------------------------
# Node mappings
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscale":             UltimateSDUpscale,
    "UltimateSDUpscaleNoUpscale":    UltimateSDUpscaleNoUpscale,
    "UltimateSDUpscaleCustomSample": UltimateSDUpscaleCustomSample,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscale":             "Ultimate SD Upscale (WAN)",
    "UltimateSDUpscaleNoUpscale":    "Ultimate SD Upscale No Upscale (WAN)",
    "UltimateSDUpscaleCustomSample": "Ultimate SD Upscale Custom Sample (WAN)",
}
