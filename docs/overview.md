---
title: Huggingface Vision
sidebar_label: Overview
sidebar_position: 1
---

The **Huggingface Vision** filter adds Hugging Face–based object detection to OpenFilter pipelines. It runs detection with `AutoImageProcessor` and `AutoModelForObjectDetection`, writes results into frame data, and optionally publishes a visualization topic with bounding boxes and labels.

The content of this document will be published to production documentation on every production release.

### ✨ Features

- **Object detection**
  - Load and run Hugging Face object detection models (e.g. RT-DETR, DETR).
  - Configurable `model_id`, `revision` (required), `threshold`, `device`, `max_detections`.
  - Results in `frame.data["subjects"]["huggingface_vision"]` with detections (label, score, box xyxy).

- **Visualization**
  - Optional topic (e.g. `viz`) with bounding boxes and labels drawn on the image, same pattern as the Protege filter.

- **Frame input**
  - Uses OpenFilter Frame convention (`frame.rw_bgr.image`); fallback to `frame.data[topic]` for custom pipelines.

### 🛠️ Use cases

- Object detection on video streams (e.g. VideoIn → FilterHuggingfaceVision → Webvis).
- Integration with other OpenFilter filters (e.g. downstream processing of detections).
- Optional viz topic for debugging or monitoring (bounding boxes on image).

### See also

- [Object detection](object-detection) — Example pipeline, variable reference, output format, and visualization.
