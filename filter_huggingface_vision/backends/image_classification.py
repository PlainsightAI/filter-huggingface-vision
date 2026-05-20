"""Image classification backend: AutoImageProcessor + AutoModelForImageClassification."""

import logging

from filter_huggingface_vision.utils import get_config_value, resolve_device

from .base import VisionBackend

logger = logging.getLogger(__name__)


def _logits_to_classifications(logits, id2label, top_k):
    """Convert model logits (batch size 1) to list of {label, score}; top_k by score descending."""
    import torch

    if logits is None or id2label is None:
        return []
    if logits.dim() != 2 or logits.shape[0] == 0:
        return []
    # Caller passes outputs.logits shape (1, num_classes); take first batch item
    logits_1d = logits[0]
    if logits_1d.numel() == 0:
        return []
    probs = torch.softmax(logits_1d, dim=-1)
    k = min(int(top_k), probs.numel())
    if k <= 0:
        return []
    scores, indices = torch.topk(probs, k, sorted=True)
    out = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        label = id2label.get(int(idx), str(int(idx)))
        out.append({"label": str(label), "score": round(float(score), 4)})
    return out


class ImageClassificationBackend(VisionBackend):
    """Backend for AutoImageProcessor + AutoModelForImageClassification (ViT, ConvNeXt, etc.)."""

    def load(self, config):
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
        )

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
            self._model = AutoModelForImageClassification.from_pretrained(
                model_id, revision=revision, trust_remote_code=False
            )
        except (ValueError, TypeError, KeyError) as e:
            raise RuntimeError(
                f"Model {model_id} (revision={revision}) is not compatible with "
                "AutoImageProcessor + AutoModelForImageClassification. "
                "Use a model supported by the Transformers image-classification API."
            ) from e

        self._model = self._model.to(self._device)
        self._model.eval()
        self._revision = revision
        logger.info(
            "ImageClassificationBackend loaded model_id=%s revision=%s device=%s",
            model_id,
            revision,
            self._device,
        )

    def run(self, image_pil, width, height, config):
        import torch

        top_k = get_config_value(config, "top_k", 5)
        top_k = max(1, min(int(top_k), 1000))

        inputs = self._processor(images=[image_pil], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits
        id2label = getattr(self._model.config, "id2label", None) or {}
        classifications = _logits_to_classifications(logits, id2label, top_k)
        return {"classifications": classifications}
