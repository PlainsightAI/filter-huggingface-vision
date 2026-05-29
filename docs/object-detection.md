---
title: Object detection
sidebar_label: Object detection
sidebar_position: 2
---

This document describes **closed-vocabulary object detection** in the filter. The filter supports several Hugging Face APIs; this one uses `AutoImageProcessor` + `AutoModelForObjectDetection`. **That API supports all models on the Hub compatible with it** (e.g. DETR, RT-DETR, Conditional DETR). See [Supported models](supported-models) for the full list of APIs and example model IDs.

## Overview

- **Object detection**: Load and run Hugging Face object detection models (e.g. `PekingU/rtdetr_r50vd`) via the `AutoImageProcessor` + `AutoModelForObjectDetection` API. Any model that loads with this API is supported.
- **Config**: `model_id`, `revision` (required), `threshold`, `device`, `max_detections`; optional visualization options (`draw_visualization`, `visualization_topic`).
- **Output**: Results are written to `frame.data["meta"]` with `detections` (list of `{class, rois}` with rois normalized [0,1]) and `detection_confidence`. Upstream meta is preserved.
- **Visualization**: When `draw_visualization=True`, the filter publishes a second topic (e.g. `viz`) with bounding boxes and labels drawn on the image.

## Example pipeline

The script `scripts/object_detection.py` runs the pipeline **VideoIn → FilterHuggingfaceVision → Webvis**. It reads configuration from the environment (e.g. a `.env` file in the project root).

### Example `.env`

```env
MODEL_ID=PekingU/rtdetr_r50vd
REVISION=main
VIDEO_PATH=./filter_example_video.mp4
THRESHOLD=0.3
```

### Run

From the project root (`.env` is loaded from the project root regardless of cwd):

```bash
python scripts/object_detection.py
```

Web UI: **http://localhost:8010**.

## Variable reference

| Variable | Required | Description |
|----------|----------|-------------|
| **MODEL_ID** | Yes | Hugging Face model id for object detection (e.g. `PekingU/rtdetr_r50vd`). Used to load the image processor and detection model. |
| **REVISION** | Yes | Model revision (e.g. `main`). Required for reproducibility; the filter validates it is non-empty. |
| **VIDEO_PATH** | Yes | Path to the input video file. Passed to VideoIn as `file://{VIDEO_PATH}!loop`. |
| **THRESHOLD** | No (default: 0.3) | Detection confidence threshold in [0, 1]. Scores below this are discarded. |
| **DRAW_VISUALIZATION** | No (default: false) | If `"true"`, the filter adds a `viz` topic with bounding boxes and labels drawn; Webvis can subscribe to both `main` and `viz`. |

## Output format

Each processed frame has `frame.data["meta"]` updated. Upstream keys (`id`, `ts`, `src`, `src_fps`) are preserved. The filter adds:

- **`detections`**: list of objects `{ "class": "<label>", "rois": [[xmin_norm, ymin_norm, xmax_norm, ymax_norm]] }` with coordinates normalized in [0, 1].
- **`detection_confidence`**: float, mean of detection scores.

Example (excerpt):

```json
{
  "id": 38,
  "ts": 1761090922.42,
  "src": "file:///path/to/video.mp4",
  "src_fps": 25.0,
  "detections": [
    { "class": "person", "rois": [[0.12, 0.19, 0.35, 0.46]] }
  ],
  "detection_confidence": 0.95
}
```

## Visualization topic

When `draw_visualization=True` (in filter config):

- The filter publishes an extra frame on the topic given by `visualization_topic` (default `"viz"`).
- The image is the same as the processed frame with bounding boxes and labels (label + score) drawn in green.
- Webvis can show both `main` and `viz` by subscribing to e.g. `tcp://localhost:5552;main,tcp://localhost:5552;viz`.

## Supported models

The filter uses Hugging Face `AutoImageProcessor` and `AutoModelForObjectDetection`, so any model supported by that API works (e.g. RT-DETR, DETR). The output of `post_process_object_detection` is normalized to the same schema whether the processor returns dict-style (`result["scores"]`) or attribute-style (`result.scores`) results.

## Renaming class names

To control the final class name shown in `meta.detections[].class` and the
visualization (e.g. report `"person"` as `"people"`, or collapse everything to a
single name), see [Class-name remapping](class-name-remapping). Works for both
closed- and open-vocabulary detection.

## See also

- [Class-name remapping](class-name-remapping) — choose the final class name shown in output and visualization.
- [Supported models — Image classification](supported-models#image-classification-vit--convnext) — ViT and ConvNeXt models for image classification (`detection_type="image-classification"`).
