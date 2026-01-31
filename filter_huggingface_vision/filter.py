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
    """Extract image and size from frame; return (PIL Image RGB, width, height) or (None, 0, 0).
    Uses OpenFilter Frame convention first (frame.rw_bgr.image, like Protege), then fallback to frame.data[topic].
    """
    try:
        from PIL import Image
        import numpy as np
        import cv2
    except ImportError:
        return None, 0, 0

    # OpenFilter Frame: image is on the frame (frame.rw_bgr.image), not in frame.data
    if getattr(frame, "has_image", False) and hasattr(frame, "rw_bgr"):
        arr = frame.rw_bgr.image  # numpy BGR (H, W, 3)
        if arr is None:
            return None, 0, 0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w = arr.shape[0], arr.shape[1]
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb.astype(np.uint8)), w, h

    # Fallback: image inside frame.data[input_topic] (e.g. custom pipelines)
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
        w, h = raw.size
        return raw.convert("RGB") if raw.mode != "RGB" else raw, w, h
    if hasattr(raw, "shape"):
        arr = np.asarray(raw)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        h, w = arr.shape[0], arr.shape[1]
        if arr.shape[-1] == 3:
            try:
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = arr
        else:
            rgb = arr
        return Image.fromarray(rgb.astype(np.uint8)), w, h

    return None, 0, 0


def _normalize_detections(results, model_config, max_detections):
    """Convert post_process_object_detection output to schema list; sort by score desc, cap at max_detections.
    Accepts both dict (e.g. RT-DETR: result['scores']) and object (e.g. result.scores) format.
    """
    out = []
    # Support both dict (RTDetrImageProcessor) and object (some other processors) output
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


def _create_visualization(image_bgr, payload):
    """Draw detection boxes and labels on a BGR image. Returns BGR numpy array."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return image_bgr.copy() if image_bgr is not None else None

    vis_image = image_bgr.copy()
    detections = payload.get("detections", [])
    for d in detections:
        label = d.get("label", "")
        score = d.get("score", 0.0)
        box = d.get("box", {})
        xmin = box.get("xmin", 0)
        ymin = box.get("ymin", 0)
        xmax = box.get("xmax", 0)
        ymax = box.get("ymax", 0)
        x1, y1, x2, y2 = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(
            vis_image, text, (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )
    return vis_image


class FilterHuggingfaceVisionConfig(FilterConfig):
    """Config for HF object detection (model_id + revision required).
    Same pattern as FilterProtegeModelConfig: class attributes only, no __init__.
    Values from variables (e.g. model_id=model_id, revision=revision in the script) are passed
    as kwargs and stored by dict.__init__(**kwargs). Class attributes below are defaults
    when the key is missing; dict_without() copy keeps all items so child process gets them.
    """

    # Visualization options (same pattern as FilterProtegeModel)
    draw_visualization: bool = False
    visualization_topic: str = "viz"
    visualization_alpha: float = 0.7
    visualization_source_topic: str = None
    # Model/config
    model_id: str = None
    revision: str = None
    task: str = "object-detection"
    threshold: float = 0.3
    device: str = "cpu"
    trust_remote_code: bool = False
    max_detections: int = 100
    input_topic: str = "main"
    output_topic: str = "main"


class FilterHuggingfaceVision(Filter):
    """
    Filter that uses Hugging Face Transformers for object detection.

    Object detection via AutoImageProcessor + AutoModelForObjectDetection.
    Supports configurable model_id, revision, threshold, and device.

    Data Signature:
    --------------
    The filter returns processed frames with the following data structure:

    Frame data:
    - Original frame data preserved
    - Processing results added to frame.data["subjects"]["huggingface_vision"]:

      - task: "object-detection"
      - model: { id, revision }
      - image: { width, height }
      - detections: list of { label, score, box: { format, xmin, ymin, xmax, ymax } }

    Visualization Frame (topic: "viz" when draw_visualization=True):
    - Image with bounding boxes and labels drawn
    - frame.data["meta"]: detections, detection_confidence

    Key Features:
    - Hugging Face object-detection models (model_id + revision required)
    - Configurable threshold and max_detections
    - Optional visualization topic (draw_visualization, visualization_topic)
    - CPU/CUDA device selection
    - JSON-serializable output (box format: xyxy)
    """

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
            draw_visualization=_get(config, "draw_visualization", False)
            if _get(config, "draw_visualization") is not None
            else _get(base, "draw_visualization", False),
            visualization_topic=_get(config, "visualization_topic", "viz")
            or _get(base, "visualization_topic", "viz"),
            visualization_alpha=_get(config, "visualization_alpha", 0.7)
            if _get(config, "visualization_alpha") is not None
            else _get(base, "visualization_alpha", 0.7),
            visualization_source_topic=_get(config, "visualization_source_topic")
            or _get(base, "visualization_source_topic"),
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
        logger.info("========= Setting up FilterHuggingfaceVision =========")
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
        except ImportError as e:
            if "timm" in str(e).lower():
                raise ImportError(
                    f"DETR/Conditional DETR models (e.g. {model_id}) require the timm library. "
                    "Install it with: pip install timm"
                ) from e
            raise
        except Exception as e:
            raise RuntimeError(
                f"Model {model_id} (revision={revision}) is not compatible with AutoImageProcessor + AutoModelForObjectDetection. "
                "Use a model supported by the Transformers object-detection API, or enable fallback when available."
            ) from e

        self._model = self._model.to(self._device)
        self._model.eval()
        self._config = config
        self._revision = revision
        self.draw_visualization = getattr(config, "draw_visualization", False)
        self.visualization_topic = getattr(config, "visualization_topic", "viz")
        self.visualization_source_topic = getattr(config, "visualization_source_topic", None)

        logger.info(
            "filter_huggingface_vision loaded model_id=%s revision=%s device=%s",
            model_id,
            revision,
            self._device,
        )

    def shutdown(self):
        logger.info("========= Shutting down FilterHuggingfaceVision =========")
        self._model = None
        self._image_processor = None
        self._config = None
        self._revision = None
        logger.info("FilterHuggingfaceVision shutdown complete.")

    def process(self, frames: dict[str, Frame]):
        import torch

        if not frames or not getattr(self, "_model", None):
            return frames

        config = self._config
        input_topic = getattr(config, "input_topic", "main")
        threshold = getattr(config, "threshold", 0.3)
        max_detections = getattr(config, "max_detections", 100)
        model_id = config.model_id

        main_frame_payload = None
        main_frame_for_viz = None

        for frame_id, frame in frames.items():
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

            if main_frame_payload is None:
                main_frame_payload = payload
                main_frame_for_viz = frame

        # Visualization topic (same pattern as FilterProtegeModel)
        if getattr(self, "draw_visualization", False) and main_frame_payload and main_frame_for_viz is not None:
            viz_topic = getattr(self, "visualization_topic", "viz")
            src_topic = getattr(self, "visualization_source_topic", None)
            image_bgr = None
            if src_topic and src_topic in frames and frames[src_topic] is not None and getattr(frames[src_topic], "has_image", False):
                image_bgr = frames[src_topic].rw_bgr.image
            if image_bgr is None and main_frame_for_viz is not None and getattr(main_frame_for_viz, "has_image", False):
                image_bgr = main_frame_for_viz.rw_bgr.image
            if image_bgr is not None:
                vis_image = _create_visualization(image_bgr, main_frame_payload)
                viz_meta = {"meta": {"detections": main_frame_payload.get("detections", []), "detection_confidence": 0.0}}
                if main_frame_payload.get("detections"):
                    scores = [d.get("score", 0) for d in main_frame_payload["detections"]]
                    viz_meta["meta"]["detection_confidence"] = sum(scores) / len(scores)
                frames[viz_topic] = Frame(vis_image, viz_meta, "BGR")

        return frames


if __name__ == "__main__":
    FilterHuggingfaceVision.run()
