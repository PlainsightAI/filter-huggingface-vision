import logging
import os
from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame

os.environ.setdefault("HF_HOME", f"{os.getcwd()}/models/hfcache")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{os.getcwd()}/models/hfcache")

__all__ = ["FilterHuggingfaceVisionConfig", "FilterHuggingfaceVision"]

logger = logging.getLogger(__name__)


# Lazy imports for transformers/torch (heavy)
def _get_processor_and_model():
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    return AutoImageProcessor, AutoModelForObjectDetection


def _image_from_frame(frame, input_topic):
    """Extract image and size from frame; return (PIL Image or tensor-friendly, width, height) or (None, 0, 0)."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None, 0, 0

    data = getattr(frame, "data", None)
    if data is None:
        return None, 0, 0

    if hasattr(data, "get"):
        raw = data.get(input_topic) or data.get("main") or data.get("image")
    else:
        raw = data

    if raw is None:
        return None, 0, 0

    if hasattr(raw, "size") and hasattr(raw, "mode"):
        # PIL Image
        w, h = raw.size
        return raw, w, h
    if hasattr(raw, "shape"):
        # numpy HWC
        arr = raw
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w = arr.shape[0], arr.shape[1]
        return Image.fromarray(arr.astype(np.uint8)), w, h

    return None, 0, 0


def _normalize_detections(results, model_config, max_detections):
    """Convert post_process_object_detection output to schema list; sort by score desc, cap at max_detections."""
    out = []
    scores = getattr(results, "scores", None)
    labels = getattr(results, "labels", None)
    boxes = getattr(results, "boxes", None)
    id2label = getattr(model_config, "id2label", None) or {}

    if scores is None or labels is None or boxes is None:
        return out

    n = min(len(scores), max_detections * 2)
    combined = list(
        zip(
            scores.tolist()[:n],
            labels.tolist()[:n],
            boxes.tolist()[:n],
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


class FilterHuggingfaceVisionConfig(FilterConfig):
    """Config for HF object detection (model_id + revision required)."""

    def __init__(
        self,
        *args,
        model_id=None,
        revision=None,
        task="object-detection",
        threshold=0.3,
        device="cpu",
        trust_remote_code=False,
        max_detections=100,
        input_topic="main",
        output_topic="main",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_id = model_id
        self.revision = revision
        self.task = task
        self.threshold = threshold
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.max_detections = max_detections
        self.input_topic = input_topic
        self.output_topic = output_topic
        # Store in dict so simpledeepcopy (used by Filter.__init__) preserves these
        self["model_id"] = model_id
        self["revision"] = revision
        self["task"] = task
        self["threshold"] = threshold
        self["device"] = device
        self["trust_remote_code"] = trust_remote_code
        self["max_detections"] = max_detections
        self["input_topic"] = input_topic
        self["output_topic"] = output_topic


class FilterHuggingfaceVision(Filter):
    """Hugging Face Vision filter: object detection via AutoImageProcessor + AutoModelForObjectDetection."""

    @classmethod
    def normalize_config(cls, config: FilterHuggingfaceVisionConfig):
        base = super().normalize_config(config)

        # Preserve our attributes: base may be dict or object and might not carry model_id/revision
        def _get(o, k, default=None):
            if hasattr(o, "get") and callable(getattr(o, "get")):
                return o.get(k, default)
            return getattr(o, k, default)

        config = FilterHuggingfaceVisionConfig(
            base,
            model_id=_get(config, "model_id") or _get(base, "model_id"),
            revision=_get(config, "revision") or _get(base, "revision"),
            task=_get(config, "task", "object-detection")
            or _get(base, "task", "object-detection"),
            threshold=_get(config, "threshold", 0.3)
            if _get(config, "threshold") is not None
            else (_get(base, "threshold", 0.3)),
            device=_get(config, "device", "cpu") or _get(base, "device", "cpu"),
            trust_remote_code=_get(config, "trust_remote_code", False)
            if _get(config, "trust_remote_code") is not None
            else _get(base, "trust_remote_code", False),
            max_detections=_get(config, "max_detections", 100)
            if _get(config, "max_detections") is not None
            else _get(base, "max_detections", 100),
            input_topic=_get(config, "input_topic", "main")
            or _get(base, "input_topic", "main"),
            output_topic=_get(config, "output_topic", "main")
            or _get(base, "output_topic", "main"),
        )

        rev = getattr(config, "revision", None)
        if rev is None or (isinstance(rev, str) and not rev.strip()):
            raise ValueError(
                "revision is required and must be non-empty (reproducibility)."
            )

        t = getattr(config, "threshold", 0.3)
        if not isinstance(t, (int, float)) or t < 0 or t > 1:
            raise ValueError("threshold must be a number in [0, 1].")

        task = getattr(config, "task", "object-detection")
        if task != "object-detection":
            raise ValueError(
                "Only task='object-detection' is supported in this version."
            )

        if getattr(config, "trust_remote_code", False):
            raise ValueError(
                "trust_remote_code=true is not allowed in this version (security)."
            )

        return config

    def setup(self, config: FilterHuggingfaceVisionConfig):
        import torch

        AutoImageProcessor, AutoModelForObjectDetection = _get_processor_and_model()

        device = getattr(config, "device", "cpu")
        if device == -1 or device == "cpu":
            self._device = torch.device("cpu")
        elif isinstance(device, int) and device >= 0:
            self._device = torch.device(
                f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(device, str) and device.startswith("cuda"):
            self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device("cpu")

        model_id = config.model_id
        revision = (config.revision or "").strip() or None
        if not revision:
            raise ValueError("revision is required and must be non-empty.")

        trust_remote_code = getattr(config, "trust_remote_code", False)

        try:
            self._image_processor = AutoImageProcessor.from_pretrained(
                model_id, revision=revision, trust_remote_code=trust_remote_code
            )
            self._model = AutoModelForObjectDetection.from_pretrained(
                model_id, revision=revision, trust_remote_code=trust_remote_code
            )
        except Exception as e:
            raise RuntimeError(
                f"Model {model_id} (revision={revision}) is not compatible with AutoImageProcessor + AutoModelForObjectDetection. "
                "Use a model supported by the Transformers object-detection API, or enable fallback when available."
            ) from e

        self._model = self._model.to(self._device)
        self._model.eval()
        self._config = config
        self._revision = revision

        logger.info(
            "filter_huggingface_vision loaded model_id=%s revision=%s device=%s",
            model_id,
            revision,
            self._device,
        )

    def shutdown(self):
        self._model = None
        self._image_processor = None
        self._config = None
        self._revision = None

    def process(self, frames: dict[str, Frame]):
        import torch

        if not frames or not getattr(self, "_model", None):
            return frames

        config = self._config
        input_topic = getattr(config, "input_topic", "main")
        threshold = getattr(config, "threshold", 0.3)
        max_detections = getattr(config, "max_detections", 100)
        model_id = config.model_id

        for frame in frames.values():
            image, width, height = _image_from_frame(frame, input_topic)
            if image is None:
                continue

            inputs = self._image_processor(images=[image], return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor([[height, width]], device=self._device)
            results = self._image_processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes
            )[0]

            detections = _normalize_detections(
                results, self._model.config, max_detections
            )
            payload = {
                "task": "object-detection",
                "model": {"id": model_id, "revision": self._revision},
                "image": {"width": width, "height": height},
                "detections": detections,
            }
            if not hasattr(frame, "data"):
                frame.data = {}
            if "subjects" not in frame.data:
                frame.data["subjects"] = {}
            frame.data["subjects"]["huggingface_vision"] = payload

        return frames


if __name__ == "__main__":
    FilterHuggingfaceVision.run()
