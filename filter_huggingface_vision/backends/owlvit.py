"""Zero-shot object detection backend: supports OWLv1 and OWLv2 via Auto classes."""

import logging
import time

from filter_huggingface_vision.utils import get_config_value, resolve_device

from ._hf_load_errors import hf_load_error_handler
from .base import VisionBackend

logger = logging.getLogger(__name__)


def _normalize_results(result, text_labels_list, max_detections):
    """Convert OWL-ViT result (boxes, scores, text_labels) to unified detection list."""
    out = []
    boxes = result.get("boxes")
    scores = result.get("scores")
    labels = result.get("text_labels")  # list of str per detection
    if boxes is None or scores is None:
        return out
    # text_labels may be added by post_process_grounded_object_detection
    if labels is None and text_labels_list and len(text_labels_list) > 0:
        label_ids = result.get("labels")  # indices
        tl = text_labels_list[0] if isinstance(text_labels_list[0], (list, tuple)) else text_labels_list
        labels = [tl[i] if i < len(tl) else str(i) for i in (label_ids.tolist() if hasattr(label_ids, "tolist") else label_ids)]
    if labels is None:
        labels = [str(i) for i in range(len(scores))]

    def _tolist(x):
        return x.tolist() if hasattr(x, "tolist") and callable(x.tolist) else list(x)

    scores_list = _tolist(scores)
    boxes_list = _tolist(boxes) if hasattr(boxes, "tolist") else list(boxes)
    n = min(len(scores_list), len(boxes_list), len(labels), max_detections)
    for i in range(n):
        score = scores_list[i]
        if not (0 <= score <= 1):
            continue
        box = boxes_list[i]
        coords = box if isinstance(box, (list, tuple)) else (box.tolist() if hasattr(box, "tolist") else list(box))
        if len(coords) < 4:
            continue
        xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
        if xmin >= xmax or ymin >= ymax:
            continue
        label = labels[i] if i < len(labels) else str(i)
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


class OwlVitBackend(VisionBackend):
    """Backend for zero-shot object detection supporting OWLv1 and OWLv2 via Auto classes."""

    def load(self, config):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._device = resolve_device(get_config_value(config, "device", "cpu"))
        model_id = get_config_value(config, "model_id")
        revision = (get_config_value(config, "revision") or "").strip() or "main"
        # Never allow trust_remote_code at load time (security); filter normalize_config rejects it, backend enforces it if used directly.
        torch_dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        with hf_load_error_handler(
            model_id, revision, "zero-shot detection (owl-vit)", "AutoProcessor"
        ):
            self._processor = AutoProcessor.from_pretrained(
                model_id, revision=revision, trust_remote_code=False
            )
        with hf_load_error_handler(
            model_id,
            revision,
            "zero-shot detection (owl-vit)",
            "AutoModelForZeroShotObjectDetection",
        ):
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, revision=revision, trust_remote_code=False, torch_dtype=torch_dtype
            )
        self._model = self._model.to(self._device)
        self._model.eval()
        self._model_dtype = torch_dtype
        self._revision = revision
        self._fps_frame_count = 0
        self._fps_start = None
        logger.info(
            "OwlVitBackend loaded model_id=%s revision=%s device=%s",
            model_id,
            revision,
            self._device,
        )

    def shutdown(self):
        if self._fps_frame_count > 0 and self._fps_start is not None:
            elapsed = time.monotonic() - self._fps_start
            fps = self._fps_frame_count / elapsed if elapsed > 0 else 0.0
            logger.info(
                "OwlVitBackend final throughput: %.2f fps (%d frames in %.1fs)",
                fps, self._fps_frame_count, elapsed,
            )
        super().shutdown()

    def run(self, image_pil, width, height, config):
        threshold = get_config_value(config, "threshold", 0.1)
        max_detections = get_config_value(config, "max_detections", 100)
        text_labels = get_config_value(config, "text_labels")
        if not text_labels or not isinstance(text_labels, (list, tuple)) or not text_labels:
            logger.warning(
                "text_labels is empty or invalid (got %r); returning zero detections for %s",
                text_labels,
                type(self).__name__,
            )
            return []
        # One image: text_labels is list of list of str, use first list for this image
        if isinstance(text_labels[0], (list, tuple)):
            text_labels_this = text_labels[0]
        else:
            text_labels_this = text_labels

        inputs = self._processor(
            text=[text_labels_this],
            images=image_pil,
            return_tensors="pt",
        )
        import torch

        inputs = {
            k: v.to(device=self._device, dtype=self._model_dtype) if hasattr(v, "to") and v.is_floating_point()
            else v.to(self._device) if hasattr(v, "to")
            else v
            for k, v in inputs.items()
        }

        if self._fps_start is None:
            self._fps_start = time.monotonic()

        with torch.no_grad():
            outputs = self._model(**inputs)

        self._fps_frame_count += 1
        target_sizes = torch.tensor([[height, width]], device=self._device)
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            text_labels=text_labels,
        )
        result = results[0] if results else {}
        return _normalize_results(result, text_labels, max_detections)
