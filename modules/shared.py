# modules/shared.py
# Full stubs for everything ultimate-upscale.py and the processing shim need.
# Covers: opts, state, sd_upscalers, actual_upscaler, batch, batch_as_tensor

sd_upscalers    = [None]
actual_upscaler = None
batch           = []
batch_as_tensor = None


class _Opts:
    """Stub for A1111's opts object.  Only the attributes used by the script."""
    img2img_background_color = "#ffffff"
    samples_format           = "png"
    upscaling_max_images_in_cache = 5

    def __getattr__(self, name):
        # Safe default for any other attribute the script may touch
        return None


class _State:
    """Stub for A1111's state object."""
    interrupted    = False
    skipped        = False
    job            = ""
    job_count      = 0
    job_no         = 0
    sampling_step  = 0
    sampling_steps = 0

    def begin(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    def nextjob(self, *args, **kwargs):
        pass


opts  = _Opts()
state = _State()
