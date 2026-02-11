---
title: Huggingface Vision
sidebar_label: Overview
sidebar_position: 1
---

The **Huggingface Vision** filter adds Hugging Face–based object detection and image classification to OpenFilter pipelines. It runs detection with `AutoImageProcessor` and `AutoModelForObjectDetection`, or classification with `AutoModelForImageClassification`, writes results into frame data, and optionally publishes a visualization topic (bounding boxes or top label).

The content of this document will be published to production documentation on every production release.

### ✨ Features

- **Object detection**
  - Load and run Hugging Face object detection models (e.g. RT-DETR, DETR).
  - Configurable `model_id`, `revision` (required), `threshold`, `device`, `max_detections`.
  - Results in `frame.data["subjects"]["huggingface_vision"]` with detections (label, score, box xyxy).

- **Image classification**
  - Load and run Hugging Face image classification models (e.g. ViT, ConvNeXt) via `detection_type="image-classification"`.
  - Configurable `model_id`, `revision` (required), `top_k`, `device`. Output `classifications` (list of label, score) in `frame.data["subjects"]["huggingface_vision"]`.

- **Visualization**
  - Optional topic (e.g. `viz`) with bounding boxes and labels (object detection) or top label + score (image classification), same pattern as the Protege filter.

- **Frame input**
  - Uses OpenFilter Frame convention (`frame.rw_bgr.image`); fallback to `frame.data[topic]` for custom pipelines.

### 🛠️ Use cases

- Object detection on video streams (e.g. VideoIn → FilterHuggingfaceVision → Webvis).
- Image classification on video streams (e.g. ViT or ConvNeXt; top-k class labels per frame).
- Integration with other OpenFilter filters (e.g. downstream processing of detections or classifications).
- Optional viz topic for debugging or monitoring (bounding boxes or top label on image).

### See also

- [Object detection](object-detection) — Example pipeline, variable reference, output format, and visualization.
- [Supported models](supported-models) — Image classification (ViT / ConvNeXt), closed-vocabulary and open-vocabulary detection models.
