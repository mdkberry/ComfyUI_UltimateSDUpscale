# usdu_nodes.py
# WAN-optimised fork of ComfyUI_UltimateSDUpscale
#
# KEY ARCHITECTURE NOTE:
# The original USDU processes ALL frames in a single script.run() call.
# shared.batch holds all PIL frames; process_batch_tiles() iterates over them.
# Our per-frame loop was calling script.run() 241 times, causing:
#   - Model re-initialisation (21s) on every frame
#   - Every debug line doubled (node loop + patch loop both running)
#   - Exponential scaling: 241 frames × 21s init = ~84 mins overhead alone
#
# Fix: pass all frames to shared.batch, call script.run() ONCE per upscale,
# and handle temporal skip + frame progress inside process_batch_tiles().

import logging
from contextlib import contextmanager

import torch
import comfy

from usdu_patch import usdu
from usdu_utils import tensor_to_pil, pil_to_tensor
from modules.processing import (
    StableDiffusionProcessing,
    clear_temporal_cache,
)
import modules.shared as shared
from modules.upscaler import UpscalerData

logger = logging.getLogger(__name__)

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


@contextmanager
def suppress_logging(level=logging.CRITICAL + 1):
    root = logging.getLogger()
    old = root.getEffectiveLevel()
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(old)


def USDU_base_inputs():
    required = [
        ("image", ("IMAGE", {"tooltip": "Image or video frame batch to upscale."})),
        ("model",         ("MODEL",        {})),
        ("positive",      ("CONDITIONING", {})),
        ("negative",      ("CONDITIONING", {})),
        ("vae",           ("VAE",          {})),
        ("upscale_by",    ("FLOAT",  {"default": 2,    "min": 0.05, "max": 4,    "step": 0.05})),
        ("seed",          ("INT",    {"default": 0,    "min": 0,    "max": 0xffffffffffffffff})),
        ("steps",         ("INT",    {"default": 1,    "min": 1,    "max": 10000, "step": 1,
                                      "tooltip": "1 step works well with LightX2V speed LoRA on WAN."})),
        ("cfg",           ("FLOAT",  {"default": 6.0,  "min": 0.0,  "max": 100.0})),
        ("sampler_name",  (comfy.samplers.KSampler.SAMPLERS, {})),
        ("scheduler",     (comfy.samplers.KSampler.SCHEDULERS, {})),
        ("denoise",       ("FLOAT",  {"default": 0.2,  "min": 0.0,  "max": 1.0,  "step": 0.01})),
        ("upscale_model", ("UPSCALE_MODEL", {})),
        ("mode_type",     (list(MODES.keys()), {})),
        ("tile_width",    ("INT",    {"default": 640,  "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("tile_height",   ("INT",    {"default": 384,  "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur",     ("INT",    {"default": 8,    "min": 0,  "max": 64,  "step": 1})),
        ("tile_padding",  ("INT",    {"default": 32,   "min": 0,  "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mode",     (list(SEAM_FIX_MODES.keys()), {})),
        ("seam_fix_denoise",  ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width",    ("INT",   {"default": 64,  "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur",("INT",   {"default": 8,   "min": 0, "max": 64,  "step": 1})),
        ("seam_fix_padding",  ("INT",   {"default": 16,  "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("force_uniform_tiles", ("BOOLEAN", {"default": True})),
        ("tiled_decode",        ("BOOLEAN", {"default": True,
                                             "tooltip": "Keep ON for WAN on 12 GB VRAM."})),
        ("batch_size",          ("INT",     {"default": 2,  "min": 1, "max": 4096, "step": 1,
                                             "tooltip": "Tiles per sampler call (spatial batching). 2 works well on 12 GB with WAN."})),
        # WAN video extras
        ("temporal_frame_skip",  ("BOOLEAN", {"default": True,
                                              "tooltip": "Skip near-identical consecutive frames."})),
        ("frame_skip_threshold", ("FLOAT",   {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.005})),
        ("seam_fix_stride",      ("INT",     {"default": 3,   "min": 1,   "max": 60,  "step": 1,
                                              "tooltip": "Apply seam fix every N frames."})),
    ]
    optional = []
    return required, optional


def prepare_inputs(required, optional=None):
    inputs = {"required": {n: t for n, t in required}}
    if optional:
        inputs["optional"] = {n: t for n, t in optional}
    return inputs


def remove_input(inputs, name):
    for i, (n, _) in enumerate(inputs):
        if n == name:
            del inputs[i]
            return


def rename_input(inputs, old, new):
    for i, (n, t) in enumerate(inputs):
        if n == old:
            inputs[i] = (new, t)
            return


# ---------------------------------------------------------------------------
# Core node
# ---------------------------------------------------------------------------

class UltimateSDUpscale:

    @classmethod
    def INPUT_TYPES(s):
        req, opt = USDU_base_inputs()
        return prepare_inputs(req, opt)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "upscale"
    CATEGORY     = "image/upscaling"
    DESCRIPTION  = (
        "WAN-optimised USDU. All frames processed in a single script.run() call "
        "to avoid per-frame model re-initialisation. Adds temporal frame skip, "
        "seam-fix stride, and GPU-native blending."
    )

    def upscale(
        self,
        image, model, positive, negative, vae,
        upscale_by, seed, steps, cfg, sampler_name, scheduler, denoise,
        upscale_model,
        mode_type, tile_width, tile_height, mask_blur, tile_padding,
        seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur, seam_fix_width, seam_fix_padding,
        force_uniform_tiles, tiled_decode, batch_size=2,
        temporal_frame_skip=True, frame_skip_threshold=0.04, seam_fix_stride=3,
        custom_sampler=None, custom_sigmas=None,
    ):
        # ---- mode enums ----
        redraw_mode     = MODES[mode_type]
        seam_fix_mode_v = SEAM_FIX_MODES[seam_fix_mode]

        # ---- upscaler stubs ----
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model

        # ---- Build frame list and put ALL frames into shared.batch ----
        # This is the critical difference vs. our broken per-frame loop.
        # The usdu_patch linear_process iterates over shared.batch internally;
        # process_batch_tiles() handles per-frame compositing.
        n_frames = image.shape[0]
        shared.batch = [tensor_to_pil(image, i) for i in range(n_frames)]
        shared.batch_as_tensor = image

        # Store WAN optimisation params on shared so process_batch_tiles can read them
        shared.temporal_frame_skip  = temporal_frame_skip
        shared.frame_skip_threshold = frame_skip_threshold
        shared.seam_fix_stride      = max(1, seam_fix_stride)
        shared.current_frame_idx    = 0      # incremented by process_batch_tiles
        shared.frame_skip_cache     = {}     # {frame_idx: PIL}

        clear_temporal_cache()

        logger.info(
            "[USDU-WAN] Starting: %d frames, tile %dx%d, batch_size=%d, "
            "temporal_skip=%s, seam_stride=%d",
            n_frames, tile_width, tile_height, batch_size,
            temporal_frame_skip, seam_fix_stride,
        )

        # ---- Build StableDiffusionProcessing for the first frame ----
        # script.run() will mutate p as it goes; shared.batch is the frame store.
        init_img = shared.batch[0]

        sdprocessing = StableDiffusionProcessing(
            init_img, model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by,
            force_uniform_tiles, tiled_decode,
            tile_width, tile_height, redraw_mode, seam_fix_mode_v,
            custom_sampler, custom_sigmas, batch_size,
            temporal_frame_skip=temporal_frame_skip,
            frame_skip_threshold=frame_skip_threshold,
            seam_fix_stride=seam_fix_stride,
        )

        # ---- Single script.run() call — processes all frames ----
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
                seams_fix_type=seam_fix_mode_v,
                target_size_type=2,
                custom_width=None, custom_height=None,
                custom_scale=upscale_by,
            )

        # ---- Collect output frames from shared.batch ----
        output_frames = shared.batch  # process_batch_tiles updated these in place
        tensors = [pil_to_tensor(f) for f in output_frames]
        return (torch.cat(tensors, dim=0),)


# ---------------------------------------------------------------------------
# No-Upscale variant
# ---------------------------------------------------------------------------

class UltimateSDUpscaleNoUpscale(UltimateSDUpscale):

    @classmethod
    def INPUT_TYPES(s):
        req, opt = USDU_base_inputs()
        remove_input(req, "upscale_model")
        remove_input(req, "upscale_by")
        rename_input(req, "image", "upscaled_image")
        return prepare_inputs(req, opt)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "upscale"
    CATEGORY     = "image/upscaling"
    DESCRIPTION  = "WAN-optimised USDU (No Upscale). Tiled i2i without prior upscaling."

    def upscale(
        self, upscaled_image, model, positive, negative, vae, seed,
        steps, cfg, sampler_name, scheduler, denoise,
        mode_type, tile_width, tile_height, mask_blur, tile_padding,
        seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur, seam_fix_width, seam_fix_padding,
        force_uniform_tiles, tiled_decode, batch_size=2,
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
        req, opt = USDU_base_inputs()
        remove_input(req, "upscale_model")
        opt.append(("upscale_model",  ("UPSCALE_MODEL", {"tooltip": "Optional; omit for Lanczos."})))
        opt.append(("custom_sampler", ("SAMPLER",  {})))
        opt.append(("custom_sigmas",  ("SIGMAS",   {})))
        return prepare_inputs(req, opt)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "upscale"
    CATEGORY     = "image/upscaling"
    DESCRIPTION  = "WAN-optimised USDU (Custom Sample)."

    def upscale(
        self, image, model, positive, negative, vae,
        upscale_by, seed, steps, cfg, sampler_name, scheduler, denoise,
        mode_type, tile_width, tile_height, mask_blur, tile_padding,
        seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur, seam_fix_width, seam_fix_padding,
        force_uniform_tiles, tiled_decode, batch_size=2,
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
