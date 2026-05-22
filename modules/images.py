# modules/images.py
# Stubs for A1111's images module used by ultimate-upscale.py.
# save_image is a no-op because ComfyUI handles its own output saving.

from PIL import Image


def flatten(img, bgcolor):
    """Replace transparency with bgcolor and return an RGB image."""
    if img.mode == "RGB":
        return img
    return Image.alpha_composite(
        Image.new("RGBA", img.size, bgcolor), img.convert("RGBA")
    ).convert("RGB")


def save_image(image, path, basename, seed=None, prompt=None,
               fmt="png", info=None, p=None, **kwargs):
    """No-op stub — ComfyUI handles saving, we don't need A1111's save logic."""
    pass
