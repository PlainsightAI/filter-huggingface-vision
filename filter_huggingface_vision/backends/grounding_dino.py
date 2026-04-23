"""Open-vocabulary object detection backend: Grounding DINO (AutoProcessor + AutoModelForZeroShotObjectDetection)."""

import logging

from filter_huggingface_vision.utils import as_bool, get_config_value, resolve_device

from ._hf_load_errors import hf_load_error_handler
from .base import VisionBackend

logger = logging.getLogger(__name__)


def _phrases_for_image(text_labels_list):
    """Configured phrases for the single image.

    text_labels_list may be nested (``[[...]]``) or flat (``[...]``) for one image;
    return the flat phrase list either way.
    """
    if not text_labels_list:
        return []
    first = text_labels_list[0]
    return first if isinstance(first, (list, tuple)) else text_labels_list


def _resolve_label(raw_label, phrases):
    """Resolve a (possibly concatenated) raw label to a single configured phrase.

    post_process_grounded_object_detection returns, per box, the union of every
    input phrase whose tokens matched (e.g. "a handgun a pistol a rifle"). Map that
    span back to one configured phrase: among phrases contained in raw_label
    (case-insensitive, whitespace-normalized, whole-word/token-boundary match), pick
    the longest (most specific), tie-breaking by configured order. The token-boundary
    check avoids false positives where one phrase is a substring of another word
    (e.g. "a cat" must not match "a caterpillar"). If none match, return raw_label
    unchanged (no silent data loss).
    """
    if not phrases:
        return raw_label
    raw_norm = " ".join(str(raw_label).split()).lower()
    # Pad with spaces so a phrase only matches on token boundaries (" a cat " is in
    # " a cat a dog " but not in " a caterpillar ").
    raw_padded = f" {raw_norm} "
    best = None
    best_len = -1
    for phrase in phrases:
        phrase_norm = " ".join(str(phrase).split()).lower()
        if phrase_norm and f" {phrase_norm} " in raw_padded and len(phrase_norm) > best_len:
            best = phrase
            best_len = len(phrase_norm)
    return best if best is not None else raw_label


def _normalize_results(result, text_labels_list, max_detections, resolve_labels=False):
    """Convert post_process_grounded_object_detection result to unified detection list.

    When ``resolve_labels`` is True, each box's (possibly concatenated) raw label is
    resolved to a single configured phrase via :func:`_resolve_label`. Off by default
    so the model's verbatim output is preserved unless the user opts in.
    """
    out = []
    boxes = result.get("boxes")
    scores = result.get("scores")
    # Grounding DINO / MM Grounding DINO may return "labels" (str) or "text_labels" (str)
    labels = result.get("text_labels") or result.get("labels")
    if boxes is None or scores is None:
        return out
    if labels is None and text_labels_list and len(text_labels_list) > 0:
        label_ids = result.get("labels")
        tl = _phrases_for_image(text_labels_list)
        if hasattr(label_ids, "tolist"):
            labels = [tl[i] if i < len(tl) else str(i) for i in label_ids.tolist()]
        elif isinstance(label_ids, (list, tuple)):
            labels = [tl[i] if i < len(tl) else str(i) for i in label_ids]
    if labels is None:
        labels = [str(i) for i in range(len(scores))]

    # Configured phrases for this image (text_labels_list may be [[...]] or a flat
    # [...] for one image; the line below handles both shapes).
    # Only populated when the user opts in; otherwise label resolution is skipped.
    phrases = _phrases_for_image(text_labels_list) if resolve_labels else []

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
        # Processor returns (x0, y0, x1, y1) in pixel coords when target_sizes is passed
        xmin = float(coords[0])
        ymin = float(coords[1])
        xmax = float(coords[2])
        ymax = float(coords[3])
        if xmin >= xmax or ymin >= ymax:
            continue
        label = labels[i] if i < len(labels) else str(i)
        label = str(label.item()) if hasattr(label, "item") else str(label)
        # Concatenated union span -> single configured phrase (opt-in via phrases).
        if phrases:
            label = _resolve_label(label, phrases)
        out.append(
            {
                "label": label,
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


class GroundingDinoBackend(VisionBackend):
    """Backend for Grounding DINO (AutoProcessor + AutoModelForZeroShotObjectDetection)."""

    def load(self, config):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._device = resolve_device(get_config_value(config, "device", "cpu"))
        model_id = get_config_value(config, "model_id")
        revision = (get_config_value(config, "revision") or "").strip() or "main"
        # Never allow trust_remote_code at load time (security); filter normalize_config rejects it, backend enforces it if used directly.
        with hf_load_error_handler(model_id, revision, "open-vocabulary detection (grounding-dino)"):
            self._processor = AutoProcessor.from_pretrained(
                model_id, revision=revision, trust_remote_code=False
            )
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id, revision=revision, trust_remote_code=False
            )
        self._model = self._model.to(self._device)
        self._model.eval()
        self._revision = revision
        logger.info(
            "GroundingDinoBackend loaded model_id=%s revision=%s device=%s",
            model_id,
            revision,
            self._device,
        )

    def run(self, image_pil, width, height, config):
        threshold = get_config_value(config, "threshold", 0.3)
        text_threshold = get_config_value(config, "text_threshold", None)
        if text_threshold is None:
            text_threshold = threshold
        else:
            text_threshold = float(text_threshold)
            if not 0.0 <= text_threshold <= 1.0:
                raise ValueError(
                    f"text_threshold must be in [0.0, 1.0]; got {text_threshold!r}"
                )
        max_detections = get_config_value(config, "max_detections", 100)
        text_labels = get_config_value(config, "text_labels")
        if not text_labels or not isinstance(text_labels, (list, tuple)) or not text_labels:
            logger.warning(
                "text_labels is empty or invalid (got %r); returning zero detections for %s",
                text_labels,
                type(self).__name__,
            )
            return []
        if isinstance(text_labels[0], (list, tuple)):
            text_labels_this = text_labels[0]
        else:
            text_labels_this = text_labels

        # Processor expects text as list of list of str (one list per image)
        text_batch = [list(text_labels_this)] if isinstance(text_labels_this, (list, tuple)) else [[str(text_labels_this)]]
        inputs = self._processor(
            images=image_pil,
            text=text_batch,
            return_tensors="pt",
        )
        import torch

        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Same API as HF example: target_sizes=[(image.height, image.width)], threshold, text_threshold
        h, w = height, width
        if hasattr(image_pil, "size") and image_pil.size:
            w, h = image_pil.size[0], image_pil.size[1]
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )
        result = results[0] if results else {}
        resolve_labels = as_bool(
            get_config_value(config, "resolve_grounding_labels", False)
        )
        return _normalize_results(result, text_labels, max_detections, resolve_labels)
