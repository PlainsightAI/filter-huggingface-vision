# Hugging Face Vision

[![PyPI version](https://img.shields.io/pypi/v/filter-huggingface-vision.svg?style=flat-square)](https://pypi.org/project/filter-huggingface-vision/)
[![Docker Version](https://img.shields.io/docker/v/plainsightai/openfilter-huggingface-vision?sort=semver)](https://hub.docker.com/r/plainsightai/openfilter-huggingface-vision)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PlainsightAI/filter-huggingface-vision/blob/main/LICENSE)

A generic filter that uses Hugging Face Transformers for vision (object detection and image classification) across video streams and OpenFilter pipelines. The filter uses one backend per **Hugging Face API**: each `detection_type` maps to a specific processor + model API. **Each API supports all models on the Hugging Face Hub that are compatible with that API**—any model loadable by the same classes will work without code changes.

### Supported Hugging Face APIs

We support the following Hugging Face APIs. Each API corresponds to one `detection_type`; each API supports **any model** from the Hub that works with that API (examples below are commonly used / tested).

| HF API (processor + model) | `detection_type` | Example model IDs |
|----------------------------|------------------|-------------------|
| `AutoImageProcessor` + `AutoModelForImageClassification` | `image-classification` | `google/vit-base-patch16-224`, `facebook/convnext-tiny-224` |
| `AutoImageProcessor` + `AutoModelForObjectDetection` | `closed-vocabulary` | `PekingU/rtdetr_r50vd`, `facebook/detr-resnet-50` |
| `OwlViTProcessor` + `OwlViTForObjectDetection` | `open-vocabulary` | `google/owlvit-base-patch32` |
| `AutoProcessor` + `AutoModelForZeroShotObjectDetection` | `open-vocabulary-grounding` | `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det` |

Full list and config examples: [docs/supported-models.md](docs/supported-models.md).

### Methods and scripts

| Method | Detection type | Script | Key config |
|--------|----------------|--------|------------|
| **Image classification** (ViT, ConvNeXt, etc.) | `image-classification` | `scripts/image_classification.py` | `MODEL_ID`, `REVISION`, `VIDEO_PATH`, optional `TOP_K` in `.env` |
| **Closed-vocabulary** (DETR, RT-DETR, Conditional DETR) | `closed-vocabulary` | `scripts/object_detection.py` | `MODEL_ID`, `REVISION`, `VIDEO_PATH` in `.env` |
| **Open-vocabulary** (OWL-ViT) | `open-vocabulary` | `scripts/zero_shot_object_detection.py` | `text_labels` in code; `VIDEO_PATH` in `.env` |
| **Open-vocabulary** (Grounding DINO) | `open-vocabulary-grounding` | `scripts/grounding_dino.py` | `text_labels` in code; `VIDEO_PATH` in `.env` |

Output is written to `frame.data["meta"]` (see [Output Structure](#output-structure)): `detections` (list of `{class, rois}` with normalized coords), `detection_confidence`; image classification also adds `classification` (`architecture`, `classes`, `confidences`).

## Features

- **Supported APIs**: Four Hugging Face APIs—image classification, closed-vocabulary object detection, OWL-ViT zero-shot, Grounding DINO. Each API supports all Hub models compatible with that API (see table above).
- **Detection types**: `image-classification`, `closed-vocabulary`, `open-vocabulary`, `open-vocabulary-grounding` via pluggable backends (one backend per API).
- **Image classification**: Run ViT, ConvNeXt, or any `AutoModelForImageClassification` model with `model_id`, `revision`, `top_k`; output `classifications` (label, score).
- **Object detection**: Run DETR, RT-DETR, etc. with `model_id`, `revision`, `threshold`, `max_detections`; output `detections` (label, score, box xyxy).
- **Zero-shot detection**: OWL-ViT or Grounding DINO with `text_labels` (list of list of str) for open-vocabulary queries.
- **Standardized output**: JSON-serializable payload in `frame.data["meta"]` (detections, detection_confidence; classification adds `meta.classification`).
- **Visualization**: Optional topic (e.g. `viz`) with bounding boxes/labels (detection) or top label + score (classification).
- **Frame input**: OpenFilter convention (`frame.rw_bgr.image`); fallback to `frame.data[topic]`.
- **Device selection**: CPU or CUDA. **Model compatibility**: Works with dict and object outputs from processors (e.g. RT-DETR, DETR).

## Architecture

The filter follows the OpenFilter pattern with three main stages:

### Stage Responsibilities

| Stage | Responsibility |
|-------|----------------|
| `setup()` | Parse and validate configuration; resolve backend by `detection_type`, load processor and model; set device |
| `process()` | Core operation: run backend inference on frame images, attach results, optionally produce visualization frame |
| `shutdown()` | Clean up resources (unload backend/model) when filter stops |

### Data Signature

The filter returns processed frames with the following data structure:

**Main Frame Data:**
- Original frame data preserved (existing `meta` keys such as `id`, `ts`, `src`, `src_fps` are kept).
- Processing results added to `frame.data["meta"]`:
  - **detections**: list of `{ class, rois }` with `rois` normalized [0,1] as `[[xmin, ymin, xmax, ymax]]`.
  - **detection_confidence**: mean score (or top score for image-classification).
  - **classification** (image-classification only): `{ architecture: "huggingface", classes: [...], confidences: [...] }`.

**Visualization Topic (when `draw_visualization=True`):**
- A separate frame is published on the configured topic (e.g. `viz`).
- Image has bounding boxes and labels drawn; `frame.data["meta"]` preserves upstream meta and includes `detections`, `detection_confidence`, and (for classification) `classification`.

## Installation

```bash
# Install with development dependencies
make install
```

## Configuration

1. Create a `.env` file in the project root (or copy from `env.example` if present).

2. Edit `.env` with your configuration:

```bash
# Required: Hugging Face model id (e.g. PekingU/rtdetr_r50vd)
MODEL_ID=PekingU/rtdetr_r50vd

# Required: Model revision (for reproducibility)
REVISION=main

# Required for pipeline script: path to input video
VIDEO_PATH=./filter_example_video.mp4

# Optional: Detection confidence threshold in [0, 1] (default: 0.3)
THRESHOLD=0.3

# Optional: Visualization (default: false)
DRAW_VISUALIZATION=true

# Optional: Webvis port (default: 8010)
PORT=8010
```

### Configuration Matrix

| Variable | Type | Default | Required | Notes |
|----------|------|---------|----------|-------|
| `model_id` | string | — | Yes | Hugging Face model id (e.g. PekingU/rtdetr_r50vd) |
| `revision` | string | — | Yes | Model revision (reproducibility) |
| `detection_type` | string | "closed-vocabulary" | No | `image-classification`, `closed-vocabulary`, `open-vocabulary`, or `open-vocabulary-grounding` |
| `top_k` | int | 5 | No | For image-classification: number of top classes to return (1–1000) |
| `text_labels` | list | — | For zero-shot / grounding | List of list of str, e.g. `[["a photo of a cat", "a photo of a dog"]]` |
| `threshold` | float | 0.3 | No | Detection confidence threshold [0, 1] (not used for image-classification) |
| `device` | string | "cpu" | No | "cpu" or "cuda" / cuda device index |
| `max_detections` | int | 100 | No | Maximum number of detections per frame (object detection only) |
| `input_topic` | string | "main" | No | Topic to read frame image from |
| `output_topic` | string | "main" | No | Topic for processed frame |
| `draw_visualization` | bool | false | No | Publish a topic with boxes/labels drawn |
| `visualization_topic` | string | "viz" | No | Topic name for visualization frame |
| `visualization_alpha` | float | 0.7 | No | Overlay alpha (reserved) |
| `visualization_source_topic` | string | — | No | Optional source topic for viz image |

## Usage

Use the script that matches your method (see table above). All scripts run VideoIn → FilterHuggingfaceVision → Webvis and serve the UI at `http://localhost:PORT` (default 8010).

### Image classification pipeline

Run image classification with a ViT, ConvNeXt, or any `AutoModelForImageClassification` model:

```bash
# In .env: MODEL_ID (e.g. google/vit-base-patch16-224 or facebook/convnext-tiny-224), REVISION=main, VIDEO_PATH, optional TOP_K
python scripts/image_classification.py
```

Output: `frame.data["meta"]` with `detections`, `detection_confidence`, and `classification` (`architecture`, `classes`, `confidences`). Visualization shows the top label + score on the image.

### Closed-vocabulary (object detection pipeline)

Run the pipeline with a fixed-vocabulary model (DETR, RT-DETR, Conditional DETR):

```bash
# Ensure MODEL_ID, REVISION, and VIDEO_PATH are set (e.g. in .env)
python scripts/object_detection.py
```

This will:
1. Load video from `VIDEO_PATH`
2. Run Hugging Face object detection on each frame (`detection_type=closed-vocabulary`)
3. Serve visualization at `http://localhost:8010` (or `PORT`); subscribe to `main` and `viz` when `DRAW_VISUALIZATION` is enabled

### Zero-shot object detection (OWL-ViT)

Run the zero-shot script (model and `text_labels` are set in the script):

```bash
# Set VIDEO_PATH in .env; edit TEXT_LABELS in scripts/zero_shot_object_detection.py if needed
python scripts/zero_shot_object_detection.py
```

Or use the filter with `detection_type="open-vocabulary"`, model `google/owlvit-base-patch32`, and `text_labels` (list of list of str):

```python
from filter_huggingface_vision.filter import FilterHuggingfaceVision, FilterHuggingfaceVisionConfig

FilterHuggingfaceVisionConfig(
    ...
    detection_type="open-vocabulary",
    model_id="google/owlvit-base-patch32",
    revision="main",
    text_labels=[["a photo of a cat", "a photo of a dog"]],
    threshold=0.1,
)
```

Output format is the same: `frame.data["meta"]` with `detections` (list of `{class, rois}` normalized), `detection_confidence`.

### Grounding DINO pipeline

Run open-vocabulary detection with Grounding DINO (model fixed in script; only `VIDEO_PATH` required in .env):

```bash
# Set VIDEO_PATH in .env (e.g. VIDEO_PATH=./filter_example_video.mp4)
python scripts/grounding_dino.py
```

See [docs/supported-models.md](docs/supported-models.md) for supported Grounding DINO model IDs and config examples.

### Using Makefile

```bash
# Run with default pipeline (from Makefile PIPELINE)
make run

# Run unit tests
make test

# Run tests with coverage
make test-coverage
```

### Visualization

When `draw_visualization=True`, the filter publishes an additional frame on the visualization topic (e.g. `viz`): bounding boxes and labels for object detection, or top label + score for image classification. Webvis subscribes to both `main` and `viz` so you can view results overlaid on the video.

## Output Structure

**Object detection** (`frame.data["meta"]`):

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

**Image classification** (`frame.data["meta"]`):

```json
{
  "id": 38,
  "ts": 1761090922.42,
  "src": "file:///path/to/video.mp4",
  "src_fps": 25.0,
  "detections": [
    { "class": "tabby cat", "rois": [[0.0, 0.0, 1.0, 1.0]] }
  ],
  "detection_confidence": 0.42,
  "classification": {
    "architecture": "huggingface",
    "classes": ["tabby cat", "Egyptian cat"],
    "confidences": [0.42, 0.31]
  }
}
```

## Development

### Project Structure

```
filter-huggingface-vision/
├── filter_huggingface_vision/
│   ├── filter.py              # Main filter implementation
│   └── backends/              # One backend per HF API (image_classification, object_detection, owlvit, grounding_dino)
├── scripts/
│   ├── image_classification.py
│   ├── object_detection.py
│   ├── zero_shot_object_detection.py
│   └── grounding_dino.py
├── docs/
│   ├── overview.md
│   ├── object-detection.md
│   └── supported-models.md
├── tests/
├── models.toml                # Model cache config (e.g. for Docker)
├── prepare_models.py          # Model prep for publish
└── pyproject.toml
```

### Key Dependencies

- `openfilter[all]~=0.1.0` - Filter framework
- `transformers>=4.40.0` - Hugging Face APIs (AutoImageProcessor + AutoModelForImageClassification / AutoModelForObjectDetection, OwlViT, AutoModelForZeroShotObjectDetection)
- `torch` - Inference
- `pillow` - Image handling
- `huggingface-hub` - Model loading
- `python-dotenv` - Environment configuration

### Testing

```bash
make test
make test-coverage
```

## Troubleshooting

### Model or revision errors
- Ensure `MODEL_ID` and `REVISION` are set. The model must be compatible with the **API** for your `detection_type`: e.g. for `image-classification` use a model that loads with `AutoModelForImageClassification` (ViT, ConvNeXt); for `closed-vocabulary` use `AutoModelForObjectDetection` (RT-DETR, DETR). See [Supported Hugging Face APIs](#supported-hugging-face-apis) and [docs/supported-models.md](docs/supported-models.md).
- Use a specific revision (e.g. `main` or a commit hash) for reproducibility.

### CUDA / device
- Set `device` to `"cpu"` if no GPU is available.
- For GPU, use `device="cuda"` or `device=0` (and ensure PyTorch is built with CUDA).

### No detections in frame
- Check that the input frame provides an image via `frame.rw_bgr.image` or `frame.data[input_topic]`.
- Lower `threshold` (e.g. 0.2) to see more detections; increase for fewer false positives.

### Visualization not showing
- Set `draw_visualization=True` in the filter config.
- Ensure Webvis (or your client) subscribes to both the main topic and the visualization topic (e.g. `viz`).

## Documentation

For more detail, pipeline examples, variable reference, and supported model IDs per method:

- [Overview](docs/overview.md)
- [Object detection](docs/object-detection.md)
- [Supported models](docs/supported-models.md)

## License

See LICENSE file for details.
