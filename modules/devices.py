# modules/devices.py
import torch


def torch_gc():
    """Free CUDA memory between tiles — important on 12 GB VRAM with WAN."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
