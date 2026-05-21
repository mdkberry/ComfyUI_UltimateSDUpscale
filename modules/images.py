# modules/images.py
from PIL import Image

def flatten(img, bgcolor):
    if img.mode == "RGB":
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert("RGB")
