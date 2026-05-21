# modules/devices.py
import torch

def torch_gc():
    """Trigger CUDA garbage collection to free memory between tiles."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
