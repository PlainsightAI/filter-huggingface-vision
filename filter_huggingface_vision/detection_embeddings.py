"""Crop embeddings for the classes the user selects.

The filter publishes JSON-ready vectors. It does not write a vector database.
A detection that fails the quality gate is left unchanged, with no embedding.
When two models are configured, both vectors are computed from that same crop
decision: either both are published, or neither is.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from filter_huggingface_vision.utils import as_bool, get_config_value


def l2_normalize(vector) -> list[float]:
    """Return a unit-length float list. A zero vector stays zero."""
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.tolist()
    return (arr / norm).tolist()


def parse_embed_classes(value):
    """Return a list of class names, or None when the setting was omitted."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            import json

            value = json.loads(text)
        else:
            value = [part.strip() for part in text.split(",") if part.strip()]
    if not isinstance(value, (list, tuple)):
        raise ValueError("embed_classes must be a list of class names.")
    names = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("embed_classes entries must be non-empty strings.")
        names.append(item.strip())
    return names


def pixel_box(detection: dict) -> tuple[float, float, float, float] | None:
    """Read the detector box in pixels. Returns None when the box is unusable."""
    box = detection.get("box") or {}
    try:
        x1 = float(box.get("xmin"))
        y1 = float(box.get("ymin"))
        x2 = float(box.get("xmax"))
        y2 = float(box.get("ymax"))
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_box(box, width, height, expansion: float):
    """Grow the box by a fraction of its size, then clamp it to the frame."""
    x1, y1, x2, y2 = box
    if expansion:
        bw = x2 - x1
        bh = y2 - y1
        x1 -= bw * expansion
        x2 += bw * expansion
        y1 -= bh * expansion
        y2 += bh * expansion
    return _clamp_box(x1, y1, x2, y2, width, height)


def square_box(box, width, height):
    """Center a square on the box. Used before a face-identity embedder."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1)
    return _clamp_box(cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0, width, height)


def touches_border(box, width, height) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= 1.0 or y1 <= 1.0 or x2 >= width - 1.0 or y2 >= height - 1.0


def sharpness(image: Image.Image) -> float:
    """Variance of a Laplacian. A blurred crop scores low."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = (
        -4.0 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def crop_passes(detection, box, crop: Image.Image, frame_size, config) -> bool:
    """True when this crop is worth storing in a vector index."""
    try:
        score = float(detection.get("score", 0.0))
    except (TypeError, ValueError):
        return False
    min_score = float(get_config_value(config, "embed_min_score", 0.5))
    if score < min_score:
        return False
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    min_area = float(get_config_value(config, "embed_min_bbox_area", 0.0))
    if area < min_area:
        return False
    width, height = frame_size
    if as_bool(get_config_value(config, "embed_reject_border", True), True) and touches_border(
        box, width, height
    ):
        return False
    min_sharp = float(get_config_value(config, "embed_min_sharpness", 0.0))
    if min_sharp > 0.0 and sharpness(crop) < min_sharp:
        return False
    return True


def crop_image(image: Image.Image, box) -> Image.Image | None:
    width, height = image.size
    clamped = _clamp_box(*box, width, height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = [int(round(v)) for v in clamped]
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def class_selected(detection, embed_classes: list[str]) -> bool:
    label = str(detection.get("label", "")).strip()
    return label in embed_classes


def attach_crop_embeddings(image, detections, config, primary, appearance) -> None:
    """Write ``embeddings`` onto detections that pass the gate.

    ``primary`` and ``appearance`` are callables ``(PIL.Image) -> vector``.
    ``appearance`` may be None. The primary crop is squared when
    ``embed_align_crops`` is set; the appearance crop is only expanded.
    """
    if not detections or primary is None:
        return
    classes = parse_embed_classes(get_config_value(config, "embed_classes", None))
    if not classes:
        return
    width, height = image.size
    expansion = float(get_config_value(config, "embed_bbox_expansion", 0.0))
    align = as_bool(get_config_value(config, "embed_align_crops", False), False)
    primary_model = str(get_config_value(config, "embedding_model_id", ""))
    appearance_model = str(get_config_value(config, "appearance_model_id", "") or "")

    for detection in detections:
        if not isinstance(detection, dict) or not class_selected(detection, classes):
            continue
        box = pixel_box(detection)
        if box is None:
            continue
        primary_box = square_box(box, width, height) if align else expand_box(box, width, height, expansion)
        appearance_box = expand_box(box, width, height, expansion)
        if primary_box is None:
            continue
        primary_crop = crop_image(image, primary_box)
        if primary_crop is None or not crop_passes(
            detection, box, primary_crop, image.size, config
        ):
            continue
        vectors = [{"model": primary_model, "vector": l2_normalize(primary(primary_crop))}]
        if appearance is not None:
            if appearance_box is None:
                continue
            appearance_crop = crop_image(image, appearance_box)
            if appearance_crop is None:
                continue
            vectors.append(
                {"model": appearance_model, "vector": l2_normalize(appearance(appearance_crop))}
            )
        detection["embeddings"] = vectors
