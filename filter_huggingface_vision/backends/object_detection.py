"""Object detection backend: AutoImageProcessor + AutoModelForObjectDetection."""

import logging

from filter_huggingface_vision.utils import get_config_value, resolve_device

from .base import VisionBackend

logger = logging.getLogger(__name__)


def _normalize_detections(results, model_config, max_detections):
    """Convert post_process_object_detection output to schema list; sort by score desc, cap at max_detections.
    Accepts both dict (e.g. RT-DETR: result['scores']) and object (e.g. result.scores) format.
    """
    out = []
    if hasattr(results, "get") and callable(getattr(results, "get")):
        scores = results.get("scores")
        labels = results.get("labels")
        boxes = results.get("boxes")
    else:
        scores = getattr(results, "scores", None)
        labels = getattr(results, "labels", None)
        boxes = getattr(results, "boxes", None)
    id2label = getattr(model_config, "id2label", None) or {}

    if scores is None or labels is None or boxes is None:
        return out

    def _tolist(x):
        return x.tolist() if hasattr(x, "tolist") and callable(x.tolist) else list(x)

    n = min(len(scores), max_detections * 2)
    combined = list(
        zip(
            _tolist(scores)[:n],
            _tolist(labels)[:n],
            _tolist(boxes)[:n],
        )
    )
    combined.sort(key=lambda x: -x[0])

    for score, label_id, box in combined[:max_detections]:
        if not (0 <= score <= 1):
            continue
        coords = box if isinstance(box, (list, tuple)) else box.tolist()
        if len(coords) >= 4:
            xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
        else:
            continue
        if xmin >= xmax or ymin >= ymax:
            continue
        label = id2label.get(int(label_id), str(int(label_id)))
        out.append(
            {
                "label": str(label),
                "score": round(float(score), 4),
                "box": {
                    "format": "xyxy",
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                },
            }
        )
    return out


class ObjectDetectionBackend(VisionBackend):
    """Backend for AutoImageProcessor + AutoModelForObjectDetection (DETR, RT-DETR, Conditional DETR, etc.)."""

    def load(self, config):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self._device = resolve_device(get_config_value(config, "device", "cpu"))
        model_id = get_config_value(config, "model_id")
        revision = (get_config_value(config, "revision") or "").strip() or None
        if not revision:
            raise ValueError("revision is required and must be non-empty.")
        # Never allow trust_remote_code at load time (security); filter normalize_config rejects it, backend enforces it if used directly.
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                model_id, revision=revision, trust_remote_code=False
            )
            self._model = AutoModelForObjectDetection.from_pretrained(
                model_id, revision=revision, trust_remote_code=False
            )
        except ImportError as e:
            if "timm" in str(e).lower():
                raise ImportError(
                    f"DETR/Conditional DETR models (e.g. {model_id}) require the timm library. "
                    "Install it with: pip install timm"
                ) from e
            raise
        except (ValueError, TypeError, KeyError) as e:
            raise RuntimeError(
                f"Model {model_id} (revision={revision}) is not compatible with AutoImageProcessor + AutoModelForObjectDetection. "
                "Use a model supported by the Transformers object-detection API, or enable fallback when available."
            ) from e

        self._model = self._model.to(self._device)
        self._model.eval()
        self._revision = revision
        logger.info(
            "ObjectDetectionBackend loaded model_id=%s revision=%s device=%s",
            model_id,
            revision,
            self._device,
        )

    def run(self, image_pil, width, height, config):
        import torch

        threshold = get_config_value(config, "threshold", 0.3)
        max_detections = get_config_value(config, "max_detections", 100)

        inputs = self._processor(images=[image_pil], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[height, width]], device=self._device)
        results = self._processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        return _normalize_detections(results, self._model.config, max_detections)
