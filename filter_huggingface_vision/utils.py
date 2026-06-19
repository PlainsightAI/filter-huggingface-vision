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
    """Resolve a config device value to a torch.device.

    - ``"cpu"`` / ``-1`` -> CPU.
    - ``"auto"`` -> CUDA when usable, otherwise CPU.
    - ``"cuda"`` / ``"cuda:N"`` / ``N`` -> that CUDA device when usable, otherwise CPU.

    When CUDA is requested but ``torch.cuda.is_available()`` is False the device falls
    back to CPU. The fallback log distinguishes a host with no GPU from a driver/PyTorch
    CUDA mismatch (a GPU is present but the installed CUDA build is too new for the
    driver) -- the common A10 failure mode -- and points at how to fix it.
    """
    import torch

    if device == -1 or device == "cpu":
        return torch.device("cpu")
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        _log_cuda_unavailable("auto", explicit=False)
        return torch.device("cpu")
    if isinstance(device, int) and device >= 0:
        if not torch.cuda.is_available():
            _log_cuda_unavailable(f"cuda:{device}")
            return torch.device("cpu")
        return torch.device(f"cuda:{device}")
    if isinstance(device, str) and device.startswith("cuda"):
        if not torch.cuda.is_available():
            _log_cuda_unavailable(device)
            return torch.device("cpu")
        return torch.device(device)
    logger.warning("Unrecognized device=%r; falling back to CPU", device)
    return torch.device("cpu")


def _log_cuda_unavailable(requested, explicit=True):
    """Log a CPU-fallback message for an unusable CUDA request.

    Distinguishes a driver/CUDA build mismatch (a GPU is present but
    ``torch.cuda.is_available()`` is False) from a host with no GPU, so the slow
    silent CPU fallback that motivated FILTER-538 becomes self-diagnosing.
    """
    import torch

    # device_count() can still report physical GPUs when is_available() is False, e.g.
    # a CUDA-13 (cu130) torch wheel on a host whose NVIDIA driver only supports CUDA 12.x.
    # That is a build/driver mismatch, not a missing GPU.
    try:
        gpu_count = torch.cuda.device_count()
    except Exception:
        gpu_count = 0

    if gpu_count > 0:
        logger.warning(
            "Requested device=%s but torch.cuda.is_available() is False while %d GPU(s) are present. "
            "The installed PyTorch CUDA build (cuda=%s) is likely incompatible with the host NVIDIA "
            "driver. Reinstall a matching build (e.g. `make install-gpu`, or `pip install torch "
            "--index-url https://download.pytorch.org/whl/cu128`) or use the official Docker image. "
            "Falling back to CPU (inference will be slow).",
            requested,
            gpu_count,
            torch.version.cuda,
        )
    elif explicit:
        logger.warning(
            "Requested device=%s but CUDA is unavailable (no GPU detected); falling back to CPU",
            requested,
        )
    else:
        logger.info("device=auto and no usable CUDA GPU detected; using CPU")
