---
title: Huggingface Vision
sidebar_label: Overview
sidebar_position: 1
---

The **Huggingface Vision** filter adds Hugging Face–based object detection and image classification to OpenFilter pipelines. The filter supports a fixed set of **Hugging Face APIs** (one per `detection_type`). **Each API supports all models on the Hugging Face Hub that are compatible with that API**—e.g. the image-classification API supports any model loadable with `AutoImageProcessor` + `AutoModelForImageClassification` (ViT, ConvNeXt, ResNet, etc.). Results are written into frame data; optionally a visualization topic is published (bounding boxes or top label).

The content of this document will be published to production documentation on every production release.

### ✨ Features

- **Supported APIs** (each supports all compatible Hub models):
  - **Image classification:** `AutoImageProcessor` + `AutoModelForImageClassification` — e.g. `google/vit-base-patch16-224`, `facebook/convnext-tiny-224`.
  - **Object detection (closed-vocabulary):** `AutoImageProcessor` + `AutoModelForObjectDetection` — e.g. `PekingU/rtdetr_r50vd`, `facebook/detr-resnet-50`.
  - **Zero-shot (OWL-ViT):** `OwlViTProcessor` + `OwlViTForObjectDetection` — e.g. `google/owlvit-base-patch32`.
  - **Zero-shot (Grounding DINO):** `AutoProcessor` + `AutoModelForZeroShotObjectDetection` — e.g. `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det`.

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
