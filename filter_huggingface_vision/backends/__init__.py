"""Registry of vision backends by detection type (closed-vocabulary / open-vocabulary)."""

from .base import VisionBackend
from .grounding_dino import GroundingDinoBackend
from .object_detection import ObjectDetectionBackend
from .owlvit import OwlVitBackend

DETECTION_TYPE_TO_BACKEND = {
    "closed-vocabulary": ObjectDetectionBackend,
    "open-vocabulary": OwlVitBackend,
    "open-vocabulary-grounding": GroundingDinoBackend,
}


def get_backend(detection_type: str):
    """Return backend class for the given detection_type. Raise ValueError if unknown."""
    if detection_type not in DETECTION_TYPE_TO_BACKEND:
        raise ValueError(
            f"Unknown detection_type '{detection_type}'. "
            f"Supported: {list(DETECTION_TYPE_TO_BACKEND.keys())}"
        )
    return DETECTION_TYPE_TO_BACKEND[detection_type]
