# modules/upscaler.py
from modules import shared

class UpscalerData:
    name = "ComfyUI Upscale Model"
    scale = 4

    def upscale(self, img, w, h):
        from usdu_utils import pil_to_tensor, tensor_to_pil
        if shared.actual_upscaler is None:
            return img.resize((w, h), resample=__import__('PIL').Image.Resampling.LANCZOS)
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        t = pil_to_tensor(img)
        (result,) = ImageUpscaleWithModel().upscale(shared.actual_upscaler, t)
        result_pil = tensor_to_pil(result, 0)
        if result_pil.size != (w, h):
            result_pil = result_pil.resize((w, h), resample=__import__('PIL').Image.Resampling.LANCZOS)
        return result_pil
