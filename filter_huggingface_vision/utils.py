"""Shared utilities for filter_huggingface_vision."""

import logging

logger = logging.getLogger(__name__)


def get_config_value(obj, key, default=None):
    """Get a value from config whether it is a dict or an object with attributes."""
    if hasattr(obj, "get") and callable(getattr(obj, "get")):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _log_cuda_diagnostics(requested: str) -> None:
    """Log CUDA availability details when a CUDA device is requested."""
    import torch

    cuda_available = torch.cuda.is_available()
    logger.info("Requested device: %s", requested)
    logger.info("CUDA available: %s", cuda_available)
    if cuda_available:
        logger.info("CUDA device count: %d", torch.cuda.device_count())
        logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))
        logger.info("CUDA version: %s", torch.version.cuda)
    else:
        logger.warning(
            "CUDA requested but not available — falling back to CPU. "
            "PyTorch built with CUDA support: %s",
            torch.backends.cuda.is_built(),
        )


def resolve_device(device):
    """Resolve config device value to a torch.device. Falls back to CPU if CUDA is unavailable."""
    import torch

    if device == -1 or device == "cpu":
        return torch.device("cpu")
    if isinstance(device, int) and device >= 0:
        requested = f"cuda:{device}"
        _log_cuda_diagnostics(requested)
        return torch.device(requested if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and device.startswith("cuda"):
        _log_cuda_diagnostics(device)
        return torch.device(device if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")
