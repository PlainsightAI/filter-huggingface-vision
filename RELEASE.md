# Changelog
Huggingface Vision filter release notes

## [Unreleased]

## v0.2.0 - 2026-02-13
### Added
- Image classification backend (AutoImageProcessor + AutoModelForImageClassification); supports ViT and ConvNeXt (e.g. `google/vit-base-patch16-224`, `facebook/convnext-tiny-224`)
- New `detection_type="image-classification"` with config `top_k`; output `classifications` (label, score) in `frame.data["subjects"]["huggingface_vision"]`
- Optional visualization for classification (top label + score as text on image)
- Script `scripts/image_classification.py` (VideoIn → FilterHuggingfaceVision → Webvis)

## v0.1.0 - 2026-02-10

### Added
- Initial release: Hugging Face Vision filter for OpenFilter
- Object detection via Hugging Face Transformers (AutoImageProcessor + AutoModelForObjectDetection)
- Configurable model_id, revision, threshold, device, and max_detections
- Output in `frame.data["subjects"]["huggingface_vision"]` with detections (label, score, box xyxy)
- Optional visualization topic with bounding boxes and labels drawn on the image
- Frame input via OpenFilter convention (`frame.rw_bgr.image`) with fallback to `frame.data[topic]`
- Object detection pipeline script (VideoIn → FilterHuggingfaceVision → Webvis)
- Support for RT-DETR and DETR-style models (dict and object detection outputs)
