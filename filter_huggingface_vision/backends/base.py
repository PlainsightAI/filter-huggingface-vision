"""Abstract base for vision backends. Each backend loads a processor/model and runs inference."""

from abc import ABC, abstractmethod


class VisionBackend(ABC):
    """Backend contract: load(config) and run(image_pil, width, height, config).
    run() may return one of:
    - A list of detections, each: { "label", "score", "box": { "format": "xyxy", "xmin", "ymin", "xmax", "ymax" } }
    - A dict with key "classifications": list of { "label": str, "score": float } (image-classification backend).
    - A dict with key "embeddings": { "embedding": list[float], "min_exemplar_distance": float } (embedding backend).
    """

    @abstractmethod
    def load(self, config):
        """Load processor and model from config. Store in self._processor, self._model, self._device.
        Raise if config is invalid or load fails.
        """
        pass

    @abstractmethod
    def run(self, image_pil, width, height, config):
        """Run inference. Return list of detections (object detection) or dict with "classifications" (image classification)."""
        pass

    def shutdown(self):
        """Release processor/model. Override if backend holds resources."""
        self._processor = None
        self._model = None
