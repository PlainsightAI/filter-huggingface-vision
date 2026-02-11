---
title: Object detection
sidebar_label: Object detection
sidebar_position: 2
---

This document describes the **first Hugging Face object detection support** in the filter: running detection with `AutoImageProcessor` and `AutoModelForObjectDetection`, writing results into frame data, and optionally publishing a visualization topic.

## Overview

- **Object detection**: Load and run Hugging Face object detection models (e.g. `PekingU/rtdetr_r50vd`) via `AutoImageProcessor` and `AutoModelForObjectDetection`.
- **Config**: `model_id`, `revision` (required), `threshold`, `device`, `max_detections`; optional visualization options (`draw_visualization`, `visualization_topic`).
- **Output**: Results are written to `frame.data["subjects"]["huggingface_vision"]` with `task`, `model`, `image`, and `detections` (label, score, box in xyxy format).
- **Visualization**: When `draw_visualization=True`, the filter publishes a second topic (e.g. `viz`) with bounding boxes and labels drawn on the image.

## Example pipeline

The script `scripts/object_detection_pipeline.py` runs the pipeline **VideoIn → FilterHuggingfaceVision → Webvis**. It reads configuration from the environment (e.g. a `.env` file in the project root).

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
python scripts/object_detection_pipeline.py
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

Each processed frame gets:

- **`frame.data["subjects"]["huggingface_vision"]`**:
  - `task`: `"object-detection"`
  - `model`: `{ "id": "<model_id>", "revision": "<revision>" }`
  - `image`: `{ "width": <int>, "height": <int> }`
  - `detections`: list of `{ "label": "<str>", "score": <float>, "box": { "format": "xyxy", "xmin", "ymin", "xmax", "ymax" } }`

## Visualization topic

When `draw_visualization=True` (in filter config):

- The filter publishes an extra frame on the topic given by `visualization_topic` (default `"viz"`).
- The image is the same as the processed frame with bounding boxes and labels (label + score) drawn in green.
- Webvis can show both `main` and `viz` by subscribing to e.g. `tcp://localhost:5552;main,tcp://localhost:5552;viz`.

## Supported models

The filter uses Hugging Face `AutoImageProcessor` and `AutoModelForObjectDetection`, so any model supported by that API works (e.g. RT-DETR, DETR). The output of `post_process_object_detection` is normalized to the same schema whether the processor returns dict-style (`result["scores"]`) or attribute-style (`result.scores`) results.

## See also

- [Supported models — Image classification](supported-models#image-classification-vit--convnext) — ViT and ConvNeXt models for image classification (`detection_type="image-classification"`).
