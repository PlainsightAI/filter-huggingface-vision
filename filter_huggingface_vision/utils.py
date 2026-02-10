"""Shared utilities for filter_huggingface_vision."""


def get_config_value(obj, key, default=None):
    """Get a value from config whether it is a dict or an object with attributes."""
    if hasattr(obj, "get") and callable(getattr(obj, "get")):
        return obj.get(key, default)
    return getattr(obj, key, default)


def resolve_device(device):
    """Resolve config device value to a torch.device. Falls back to CPU if CUDA is unavailable."""
    import torch

    if device == -1 or device == "cpu":
        return torch.device("cpu")
    if isinstance(device, int) and device >= 0:
        return torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
    if isinstance(device, str) and device.startswith("cuda"):
        return torch.device(device if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")
