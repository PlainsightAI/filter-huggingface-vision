"""Zero-shot object detection backend: OwlViTProcessor + OwlViTForObjectDetection."""

import logging

from filter_huggingface_vision.utils import get_config_value, resolve_device

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
    """Backend for OwlViTProcessor + OwlViTForObjectDetection (zero-shot with text_labels)."""

    def load(self, config):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self._device = resolve_device(get_config_value(config, "device", "cpu"))
        model_id = get_config_value(config, "model_id")
        revision = (get_config_value(config, "revision") or "").strip() or None
        if not revision:
            raise ValueError("revision is required and must be non-empty.")
        # Never allow trust_remote_code at load time (security); filter normalize_config rejects it, backend enforces it if used directly.
        self._processor = OwlViTProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        self._model = OwlViTForObjectDetection.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        self._revision = revision
        logger.info(
            "OwlVitBackend loaded model_id=%s revision=%s device=%s",
            model_id,
            revision,
            self._device,
        )

    def run(self, image_pil, width, height, config):
        threshold = get_config_value(config, "threshold", 0.1)
        max_detections = get_config_value(config, "max_detections", 100)
        text_labels = get_config_value(config, "text_labels")
        if not text_labels or not isinstance(text_labels, (list, tuple)) or not text_labels:
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

        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[height, width]], device=self._device)
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            text_labels=text_labels,
        )
        result = results[0] if results else {}
        return _normalize_results(result, text_labels, max_detections)
