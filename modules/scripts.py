# modules/scripts.py
# Stub for A1111's scripts module.
# ultimate-upscale.py inherits from scripts.Script; we provide a minimal base class.


class Script:
    """Minimal A1111 Script base class stub."""

    def title(self):
        return ""

    def show(self, is_img2img):
        return False

    def ui(self, is_img2img):
        return []

    def run(self, *args, **kwargs):
        pass
