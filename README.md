# Hugging Face Vision

[![PyPI version](https://img.shields.io/pypi/v/filter-huggingface-vision.svg?style=flat-square)](https://pypi.org/project/filter-huggingface-vision/)
[![Docker Version](https://img.shields.io/docker/v/plainsightai/openfilter-huggingface-vision?sort=semver)](https://hub.docker.com/r/plainsightai/openfilter-huggingface-vision)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PlainsightAI/filter-huggingface-vision/blob/main/LICENSE)

A generic filter that uses Hugging Face Transformers for vision tasks (object detection, zero-shot object detection) across video streams and OpenFilter pipelines. The filter uses a backend per task so multiple model interfaces (AutoImageProcessor, OwlViTProcessor, etc.) are supported with a single config and unified output format.

## Features

- **Multiple tasks**: `object-detection` (DETR, RT-DETR, Conditional DETR) and `zero-shot-object-detection` (OWL-ViT) via pluggable backends
- **Object Detection**: Run Hugging Face object-detection models with configurable `model_id`, `revision`, `threshold`, and `max_detections`
- **Zero-shot detection**: OWL-ViT with `text_labels` (list of list of str) for open-vocabulary queries
- **Standardized Output**: JSON-serializable detections with label, score, and box (xyxy format) in `frame.data["subjects"]["huggingface_vision"]`
- **Visualization**: Optional topic (e.g. `viz`) with bounding boxes and labels drawn on the image
- **Frame Input**: Uses OpenFilter Frame convention (`frame.rw_bgr.image`); fallback to `frame.data[topic]` for custom pipelines
- **Device Selection**: CPU or CUDA
- **Pipeline Integration**: Works with OpenFilter pipeline architecture (VideoIn → FilterHuggingfaceVision → Webvis)
- **Environment Configuration**: Configuration via environment variables or config dict
- **Model Compatibility**: Supports both dict and object outputs from `post_process_object_detection` (e.g. RT-DETR and DETR-style processors)

## Architecture

The filter follows the OpenFilter pattern with three main stages:

### Stage Responsibilities

| Stage | Responsibility |
|-------|----------------|
| `setup()` | Parse and validate configuration; resolve backend by `task`, load processor and model; set device |
| `process()` | Core operation: run backend inference on frame images, attach results, optionally produce visualization frame |
| `shutdown()` | Clean up resources (unload backend/model) when filter stops |

### Data Signature

The filter returns processed frames with the following data structure:

**Main Frame Data:**
- Original frame data preserved
- Processing results added to `frame.data["subjects"]["huggingface_vision"]`:
  - `task`: `"object-detection"` or `"zero-shot-object-detection"`
  - `model`: `{ "id": "<model_id>", "revision": "<revision>" }`
  - `image`: `{ "width": int, "height": int }`
  - `detections`: list of `{ "label": str, "score": float, "box": { "format": "xyxy", "xmin", "ymin", "xmax", "ymax" } }`

**Visualization Topic (when `draw_visualization=True`):**
- A separate frame is published on the configured topic (e.g. `viz`)
- Image has bounding boxes and labels drawn; `frame.data["meta"]` includes `detections` and `detection_confidence`

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
| `task` | string | "object-detection" | No | `object-detection` or `zero-shot-object-detection` |
| `text_labels` | list | — | For zero-shot | List of list of str, e.g. `[["a photo of a cat", "a photo of a dog"]]` |
| `threshold` | float | 0.3 | No | Detection confidence threshold [0, 1] |
| `device` | string | "cpu" | No | "cpu" or "cuda" / cuda device index |
| `max_detections` | int | 100 | No | Maximum number of detections per frame |
| `input_topic` | string | "main" | No | Topic to read frame image from |
| `output_topic` | string | "main" | No | Topic for processed frame |
| `draw_visualization` | bool | false | No | Publish a topic with boxes/labels drawn |
| `visualization_topic` | string | "viz" | No | Topic name for visualization frame |
| `visualization_alpha` | float | 0.7 | No | Overlay alpha (reserved) |
| `visualization_source_topic` | string | — | No | Optional source topic for viz image |

## Usage

### Object Detection Pipeline

Run the pipeline (VideoIn → FilterHuggingfaceVision → Webvis):

```bash
# Ensure MODEL_ID, REVISION, and VIDEO_PATH are set (e.g. in .env)
python scripts/object_detection_pipeline.py
```

This will:
1. Load video from `VIDEO_PATH`
2. Run Hugging Face object detection on each frame (task=`object-detection`)
3. Serve visualization at `http://localhost:8010` (or `PORT`); subscribe to `main` and `viz` when `DRAW_VISUALIZATION` is enabled

### Zero-shot object detection (OWL-ViT)

Use task `zero-shot-object-detection` with a model like `google/owlvit-base-patch32` and provide `text_labels` (list of list of str) for open-vocabulary queries:

```python
from filter_huggingface_vision.filter import FilterHuggingfaceVision, FilterHuggingfaceVisionConfig

FilterHuggingfaceVisionConfig(
    ...
    task="zero-shot-object-detection",
    model_id="google/owlvit-base-patch32",
    revision="main",
    text_labels=[["a photo of a cat", "a photo of a dog"]],
    threshold=0.1,
)
```

Output format is the same: `frame.data["subjects"]["huggingface_vision"]` with `task`, `model`, `image`, and `detections` (label, score, box).

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

When `draw_visualization=True`, the filter publishes an additional frame on the visualization topic (e.g. `viz`) with bounding boxes and labels drawn. In the pipeline script, Webvis is configured to subscribe to both `main` and `viz` so you can view detections overlaid on the video.

## Output Structure

**Frame payload** (`frame.data["subjects"]["huggingface_vision"]`):

```json
{
  "task": "object-detection",
  "model": { "id": "PekingU/rtdetr_r50vd", "revision": "main" },
  "image": { "width": 1920, "height": 1080 },
  "detections": [
    {
      "label": "person",
      "score": 0.95,
      "box": { "format": "xyxy", "xmin": 100, "ymin": 200, "xmax": 300, "ymax": 500 }
    }
  ]
}
```

## Development

### Project Structure

```
filter-huggingface-vision/
├── filter_huggingface_vision/
│   └── filter.py              # Main filter implementation
├── scripts/
│   └── object_detection_pipeline.py
├── docs/
│   ├── overview.md
│   └── object-detection.md
├── tests/
├── models.toml                # Model cache config (e.g. for Docker)
├── prepare_models.py          # Model prep for publish
└── pyproject.toml
```

### Key Dependencies

- `openfilter[all]~=0.1.0` - Filter framework
- `transformers>=4.40.0` - Hugging Face AutoImageProcessor / AutoModelForObjectDetection
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
- Ensure `MODEL_ID` and `REVISION` are set and the model is compatible with `AutoImageProcessor` + `AutoModelForObjectDetection` (e.g. RT-DETR, DETR).
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

For more detail, pipeline examples, and variable reference, see:

- [Overview](docs/overview.md)
- [Object detection](docs/object-detection.md)

## License

See LICENSE file for details.
