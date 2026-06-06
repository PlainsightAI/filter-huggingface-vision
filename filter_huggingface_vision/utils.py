"""Shared utilities for filter_huggingface_vision."""

import logging

logger = logging.getLogger(__name__)


def get_config_value(obj, key, default=None):
    """Get a value from config whether it is a dict or an object with attributes."""
    if hasattr(obj, "get") and callable(getattr(obj, "get")):
        return obj.get(key, default)
    return getattr(obj, key, default)


def as_bool(value, default=False):
    """Coerce a config value to bool, accepting the string forms env vars produce.

    OpenFilter passes env-sourced flags as strings, so a plain ``bool("false")``
    would be truthy. Treat ``1``/``true``/``yes``/``on`` (case-insensitive) as True
    and everything else as False; ``None`` returns ``default``.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def resolve_device(device):
    """Resolve config device value to a torch.device. Falls back to CPU if CUDA is unavailable."""
    import torch

    if device == -1 or device == "cpu":
        return torch.device("cpu")
    if isinstance(device, int) and device >= 0:
        if not torch.cuda.is_available():
            logger.warning("Requested device=cuda:%s but CUDA is unavailable; falling back to CPU", device)
            return torch.device("cpu")
        return torch.device(f"cuda:{device}")
    if isinstance(device, str) and device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("Requested device=%s but CUDA is unavailable; falling back to CPU", device)
            return torch.device("cpu")
        return torch.device(device)
    logger.warning("Unrecognized device=%r; falling back to CPU", device)
    return torch.device("cpu")
