import logging
import os
from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame

from filter_huggingface_vision.backends import get_backend
from filter_huggingface_vision.utils import get_config_value

os.environ.setdefault("HF_HOME", f"{os.getcwd()}/models/hfcache")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{os.getcwd()}/models/hfcache")

__all__ = ["FilterHuggingfaceVisionConfig", "FilterHuggingfaceVision"]

logger = logging.getLogger(__name__)


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
        x1, y1, x2, y2 = (
            int(round(xmin)),
            int(round(ymin)),
            int(round(xmax)),
            int(round(ymax)),
        )
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(
            vis_image,
            text,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
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
    detection_type: str = "closed-vocabulary"  # "closed-vocabulary" | "open-vocabulary"
    threshold: float = 0.3
    device: str = "cpu"
    trust_remote_code: bool = False
    max_detections: int = 100
    input_topic: str = "main"
    output_topic: str = "main"
    # Zero-shot (e.g. OWL-ViT): list of list of str, e.g. [["a photo of a cat", "a photo of a dog"]]
    text_labels: list = None


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

      - detection_type: "closed-vocabulary" | "open-vocabulary" (task kept for backward compat)
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
        config = FilterHuggingfaceVisionConfig(
            base,
            draw_visualization=get_config_value(config, "draw_visualization", False)
            if get_config_value(config, "draw_visualization") is not None
            else get_config_value(base, "draw_visualization", False),
            visualization_topic=get_config_value(config, "visualization_topic", "viz")
            or get_config_value(base, "visualization_topic", "viz"),
            visualization_alpha=get_config_value(config, "visualization_alpha", 0.7)
            if get_config_value(config, "visualization_alpha") is not None
            else get_config_value(base, "visualization_alpha", 0.7),
            visualization_source_topic=get_config_value(
                config, "visualization_source_topic"
            )
            or get_config_value(base, "visualization_source_topic"),
            model_id=get_config_value(config, "model_id")
            or get_config_value(base, "model_id"),
            revision=get_config_value(config, "revision")
            or get_config_value(base, "revision"),
            detection_type=get_config_value(
                config, "detection_type", "closed-vocabulary"
            )
            or get_config_value(base, "detection_type", "closed-vocabulary"),
            threshold=get_config_value(config, "threshold", 0.3)
            if get_config_value(config, "threshold") is not None
            else (get_config_value(base, "threshold", 0.3)),
            device=get_config_value(config, "device", "cpu")
            or get_config_value(base, "device", "cpu"),
            trust_remote_code=get_config_value(config, "trust_remote_code", False)
            if get_config_value(config, "trust_remote_code") is not None
            else get_config_value(base, "trust_remote_code", False),
            max_detections=get_config_value(config, "max_detections", 100)
            if get_config_value(config, "max_detections") is not None
            else get_config_value(base, "max_detections", 100),
            input_topic=get_config_value(config, "input_topic", "main")
            or get_config_value(base, "input_topic", "main"),
            output_topic=get_config_value(config, "output_topic", "main")
            or get_config_value(base, "output_topic", "main"),
            text_labels=get_config_value(config, "text_labels")
            or get_config_value(base, "text_labels"),
        )

        rev = getattr(config, "revision", None)
        if rev is None or (isinstance(rev, str) and not rev.strip()):
            raise ValueError(
                "revision is required and must be non-empty (reproducibility)."
            )

        t = getattr(config, "threshold", 0.3)
        if not isinstance(t, (int, float)) or t < 0 or t > 1:
            raise ValueError("threshold must be a number in [0, 1].")

        detection_type = getattr(config, "detection_type", "closed-vocabulary")
        get_backend(detection_type)  # validate detection_type is registered

        if (
            detection_type == "open-vocabulary"
            or detection_type == "open-vocabulary-grounding"
        ):
            tl = get_config_value(config, "text_labels") or get_config_value(
                base, "text_labels"
            )
            if not tl or not isinstance(tl, (list, tuple)) or not tl:
                raise ValueError(
                    f"detection_type='{detection_type}' requires text_labels "
                    "(list of list of str, e.g. [['a photo of a cat', 'a photo of a dog']])."
                )
            if not isinstance(tl[0], (list, tuple)):
                raise ValueError(
                    "text_labels must be list of list of str, e.g. [['cat', 'dog']]."
                )

        if getattr(config, "trust_remote_code", False):
            raise ValueError(
                "trust_remote_code=true is not allowed in this version (security)."
            )

        return config

    def setup(self, config: FilterHuggingfaceVisionConfig):
        logger.info("========= Setting up FilterHuggingfaceVision =========")
        detection_type = getattr(config, "detection_type", "closed-vocabulary")
        backend_class = get_backend(detection_type)
        self._backend = backend_class()
        self._backend.load(config)
        self._config = config
        self._revision = (getattr(config, "revision") or "").strip() or None
        self.draw_visualization = getattr(config, "draw_visualization", False)
        self.visualization_topic = getattr(config, "visualization_topic", "viz")
        self.visualization_source_topic = getattr(
            config, "visualization_source_topic", None
        )
        logger.info(
            "filter_huggingface_vision loaded detection_type=%s model_id=%s revision=%s",
            detection_type,
            getattr(config, "model_id"),
            self._revision,
        )

    def shutdown(self):
        logger.info("========= Shutting down FilterHuggingfaceVision =========")
        if getattr(self, "_backend", None) is not None:
            self._backend.shutdown()
        self._backend = None
        self._config = None
        self._revision = None
        logger.info("FilterHuggingfaceVision shutdown complete.")

    def process(self, frames: dict[str, Frame]):
        if not frames or not getattr(self, "_backend", None):
            return frames

        # So Webvis (sources="...;main,...;viz") always gets a "main" stream: if the
        # upstream sent a single frame under another key, treat it as "main".
        output_topic = getattr(self._config, "output_topic", "main")
        if output_topic not in frames:
            only_key = next(iter(frames))
            frames[output_topic] = frames.pop(only_key)

        config = self._config
        input_topic = getattr(config, "input_topic", "main")
        detection_type = getattr(config, "detection_type", "closed-vocabulary")
        model_id = getattr(config, "model_id", "")

        main_frame_payload = None
        main_frame_for_viz = None

        for frame_id, frame in frames.items():
            image, width, height = _image_from_frame(frame, input_topic)
            if image is None:
                continue

            detections = self._backend.run(image, width, height, config)
            # task: legacy key for backward compatibility (same values as before)
            _task = (
                "object-detection"
                if detection_type == "closed-vocabulary"
                else "zero-shot-object-detection"
            )
            payload = {
                "detection_type": detection_type,
                "task": _task,
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
        if (
            getattr(self, "draw_visualization", False)
            and main_frame_payload
            and main_frame_for_viz is not None
        ):
            viz_topic = getattr(self, "visualization_topic", "viz")
            src_topic = getattr(self, "visualization_source_topic", None)
            image_bgr = None
            if (
                src_topic
                and src_topic in frames
                and frames[src_topic] is not None
                and getattr(frames[src_topic], "has_image", False)
            ):
                image_bgr = frames[src_topic].rw_bgr.image
            if (
                image_bgr is None
                and main_frame_for_viz is not None
                and getattr(main_frame_for_viz, "has_image", False)
            ):
                image_bgr = main_frame_for_viz.rw_bgr.image
            if image_bgr is not None:
                vis_image = _create_visualization(image_bgr, main_frame_payload)
                viz_meta = {
                    "meta": {
                        "detections": main_frame_payload.get("detections", []),
                        "detection_confidence": 0.0,
                    }
                }
                if main_frame_payload.get("detections"):
                    scores = [
                        d.get("score", 0) for d in main_frame_payload["detections"]
                    ]
                    viz_meta["meta"]["detection_confidence"] = sum(scores) / len(scores)
                frames[viz_topic] = Frame(vis_image, viz_meta, "BGR")

        return frames


if __name__ == "__main__":
    FilterHuggingfaceVision.run()
