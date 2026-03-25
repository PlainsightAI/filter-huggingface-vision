"""Shared utilities for filter_huggingface_vision."""

import logging

logger = logging.getLogger(__name__)


def get_config_value(obj, key, default=None):
    """Get a value from config whether it is a dict or an object with attributes."""
    if hasattr(obj, "get") and callable(getattr(obj, "get")):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _log_cuda_info(requested: str) -> None:
    """Log CUDA availability info when CUDA is available and requested."""
    import torch

    logger.info("Requested device: %s", requested)
    logger.info("CUDA available: True")
    logger.info("CUDA device count: %d", torch.cuda.device_count())
    logger.info("CUDA device name: %s", torch.cuda.get_device_name(0))
    logger.info("CUDA version: %s", torch.version.cuda)


def _raise_cuda_unavailable(requested: str) -> None:
    """Raise RuntimeError with diagnostics when CUDA is explicitly requested but unavailable."""
    import torch

    pytorch_cuda = torch.version.cuda or "none"
    is_built = torch.backends.cuda.is_built()
    raise RuntimeError(
        f"FILTER_DEVICE={requested!r} was explicitly requested but CUDA is not available. "
        f"PyTorch CUDA version: {pytorch_cuda}. "
        f"PyTorch built with CUDA support: {is_built}. "
        "Check that your NVIDIA driver is installed and meets the minimum version required "
        f"for CUDA {pytorch_cuda}, then verify with 'nvidia-smi'."
    )


def resolve_device(device: "int | str") -> "torch.device":
    """Resolve config device value to a torch.device.

    - ``cpu`` / ``-1``: always use CPU, no CUDA check.
    - ``auto``: use CUDA if available, otherwise warn and fall back to CPU.
    - ``cuda`` / ``cuda:N`` / non-negative integer: use CUDA; raise RuntimeError if unavailable.
    """
    import torch

    if device == -1 or device == "cpu":
        return torch.device("cpu")

    if device == "auto":
        if torch.cuda.is_available():
            logger.info("FILTER_DEVICE=auto: CUDA available, using cuda")
            return torch.device("cuda")
        logger.warning(
            "FILTER_DEVICE=auto: CUDA not available -- falling back to CPU. "
            "PyTorch built with CUDA support: %s",
            torch.backends.cuda.is_built(),
        )
        return torch.device("cpu")

    if isinstance(device, int) and device >= 0:
        requested = f"cuda:{device}"
        if torch.cuda.is_available():
            _log_cuda_info(requested)
            return torch.device(requested)
        _raise_cuda_unavailable(requested)

    if isinstance(device, str) and device.startswith("cuda"):
        if torch.cuda.is_available():
            _log_cuda_info(device)
            return torch.device(device)
        _raise_cuda_unavailable(device)

    return torch.device("cpu")
