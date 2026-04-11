---
title: Huggingface Vision
sidebar_label: Overview
sidebar_position: 1
---

The **Huggingface Vision** filter adds Hugging Face–based object detection, image classification, and embedding extraction to OpenFilter pipelines. The filter supports a fixed set of **Hugging Face APIs** (one per `detection_type`). **Each API supports all models on the Hugging Face Hub that are compatible with that API**—e.g. the image-classification API supports any model loadable with `AutoImageProcessor` + `AutoModelForImageClassification` (ViT, ConvNeXt, ResNet, etc.). The embedding backend is model-agnostic: it uses PyTorch forward hooks to extract penultimate-layer features from any vision model. Results are written into frame data; optionally a visualization topic is published (bounding boxes or top label).

The content of this document will be published to production documentation on every production release.

The **official Docker image** on `linux/amd64` ships PyTorch built for **CUDA 12.8** so GPU deployments stay compatible with common driver caps; local PyPI installs are unchanged.

### ✨ Features

- **Supported APIs** (each supports all compatible Hub models):
  - **Image classification:** `AutoImageProcessor` + `AutoModelForImageClassification` — e.g. `google/vit-base-patch16-224`, `facebook/convnext-tiny-224`.
  - **Object detection (closed-vocabulary):** `AutoImageProcessor` + `AutoModelForObjectDetection` — e.g. `PekingU/rtdetr_r50vd`, `facebook/detr-resnet-50`.
  - **Zero-shot (OWL-ViT):** `OwlViTProcessor` + `OwlViTForObjectDetection` — e.g. `google/owlvit-base-patch32`.
  - **Zero-shot (Grounding DINO):** `AutoProcessor` + `AutoModelForZeroShotObjectDetection` — e.g. `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det`.
  - **Embedding extraction:** Any `AutoModel` / `AutoModelFor*` (via hook) / timm model — e.g. `facebook/dinov2-small`, `google/vit-base-patch16-224`, `convnext_tiny.dinov3_lvd1689m` (timm).

- **Object detection**
  - Load and run Hugging Face object detection models (e.g. RT-DETR, DETR).
  - Configurable `model_id`, `revision` (required), `threshold`, `device`, `max_detections`.
  - Results in `frame.data["meta"]` with `detections` (list of `{class, rois}` normalized), `detection_confidence`.

- **Image classification**
  - Load and run Hugging Face image classification models (e.g. ViT, ConvNeXt) via `detection_type="image-classification"`.
  - Configurable `model_id`, `revision` (required), `top_k`, `device`. Output in `frame.data["meta"]` with `detection_type`, `task`, `model`, and `classification` (`architecture`, `classes`, `confidences`). No `detections` or `detection_confidence` for classification.

- **Embedding extraction**
  - Extract penultimate-layer feature embeddings from any vision model via `detection_type="embedding"`.
  - Model-agnostic: uses PyTorch forward hooks to capture the last representation before the output head. Works with classification, detection, segmentation, and pure feature extractor models.
  - Supports HuggingFace Transformers and timm via the `model_loader` config option.
  - Optional exemplar distance: provide a `.npz` file of reference embeddings to get `min_exemplar_distance` (L2) per frame for similarity-based anomaly detection.
  - Output in `frame.data["embedding"]` (feature vector) and optionally `frame.data["min_exemplar_distance"]`.

- **Visualization**
  - Optional topic (e.g. `viz`) with bounding boxes and labels (object detection) or top label + score (image classification), same pattern as the Protege filter.

- **Frame input**
  - Uses OpenFilter Frame convention (`frame.rw_bgr.image`); fallback to `frame.data[topic]` for custom pipelines.

### 🛠️ Use cases

- Object detection on video streams (e.g. VideoIn → FilterHuggingfaceVision → Webvis).
- Image classification on video streams (e.g. ViT or ConvNeXt; top-k class labels per frame).
- Embedding extraction for similarity search, anomaly detection, or downstream ML (any vision model).
- Integration with other OpenFilter filters (e.g. downstream processing of detections, classifications, or embeddings).
- Optional viz topic for debugging or monitoring (bounding boxes or top label on image).

### See also

- [Object detection](object-detection) — Example pipeline, variable reference, output format, and visualization.
- [Supported models](supported-models) — Image classification (ViT / ConvNeXt), closed-vocabulary and open-vocabulary detection models.
