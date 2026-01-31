"""Registry of vision backends by task."""

from .base import VisionBackend
from .object_detection import ObjectDetectionBackend
from .owlvit import OwlVitBackend

TASK_TO_BACKEND = {
    "object-detection": ObjectDetectionBackend,
    "zero-shot-object-detection": OwlVitBackend,
}


def get_backend(task: str):
    """Return backend class for the given task. Raise ValueError if task is unknown."""
    if task not in TASK_TO_BACKEND:
        raise ValueError(
            f"Unknown task '{task}'. Supported: {list(TASK_TO_BACKEND.keys())}"
        )
    return TASK_TO_BACKEND[task]
