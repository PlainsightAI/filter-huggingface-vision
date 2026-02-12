# Supported models by detection type

The filter supports a fixed set of **Hugging Face APIs**. Each API is identified by a processor + model class pair and corresponds to one `detection_type`. **Each API supports all models on the Hugging Face Hub that are compatible with that API**—any model that can be loaded with the same processor and model classes will work. Below we list each supported API and **example model IDs** (tested or commonly used); the Hub may list additional compatible models for each API.

## Supported APIs (summary)

| HF API (processor + model) | `detection_type` | Example model IDs |
|----------------------------|------------------|-------------------|
| `AutoImageProcessor` + `AutoModelForImageClassification` | `image-classification` | `google/vit-base-patch16-224`, `facebook/convnext-tiny-224` |
| `AutoImageProcessor` + `AutoModelForObjectDetection` | `closed-vocabulary` | `PekingU/rtdetr_r50vd`, `facebook/detr-resnet-50` |
| `OwlViTProcessor` + `OwlViTForObjectDetection` | `open-vocabulary` | `google/owlvit-base-patch32` |
| `AutoProcessor` + `AutoModelForZeroShotObjectDetection` | `open-vocabulary-grounding` | `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det` |

Output is written to `frame.data["subjects"]["huggingface_vision"]`: for object detection use `detections` (label, score, box xyxy); for image classification use `classifications` (label, score).

## Pipelines

| Script | Detection type | Model source |
|--------|----------------|--------------|
| `scripts/object_detection.py` | Closed-vocabulary (DETR / RT-DETR) | `MODEL_ID` + `REVISION` from .env |
| `scripts/image_classification.py` | Image classification (ViT / ConvNeXt) | `MODEL_ID` + `REVISION` from .env |
| `scripts/zero_shot_object_detection.py` | Open-vocabulary (OWL-ViT) | Fixed in code: `google/owlvit-base-patch32` @ main |
| `scripts/grounding_dino.py` | Open-vocabulary (Grounding DINO) | Fixed in code: MM Grounding DINO tiny @ main |

The zero-shot and Grounding DINO scripts use a fixed model in code so the same .env (e.g. with `VIDEO_PATH`) can be shared without loading the wrong model.

---

## Image classification (ViT / ConvNeXt)

**API:** `AutoImageProcessor` + `AutoModelForImageClassification`. Any model on the Hub that loads with this API is supported. Assigns one or more class labels to an image (e.g. ImageNet classes). No `text_labels` required; optional `top_k` (default 5) controls how many top classes to return.

**Example model IDs:**

| MODEL_ID | REVISION |
|----------|----------|
| google/vit-base-patch16-224 | main |
| facebook/convnext-tiny-224 | main |

### Example .env (image classification pipeline)

```bash
MODEL_ID=google/vit-base-patch16-224
REVISION=main
VIDEO_PATH=./filter_example_video.mp4
TOP_K=5
PORT=8010
```

### Example config (image-classification, in code)

```python
FilterHuggingfaceVisionConfig(
    detection_type="image-classification",
    model_id="google/vit-base-patch16-224",
    revision="main",
    top_k=5,
)
```

---

## Closed-vocabulary object detection (DETR / RT-DETR)

**API:** `AutoImageProcessor` + `AutoModelForObjectDetection`. Any model on the Hub that loads with this API is supported (e.g. DETR, RT-DETR, Conditional DETR). Fixed set of classes (e.g. COCO).  
**Dependency:** `timm` (for DETR/Conditional DETR backbones).

**Example model IDs:**

| MODEL_ID | REVISION |
|----------|----------|
| facebook/detr-resnet-50 | main |
| microsoft/conditional-detr-resnet-50 | main |
| PekingU/rtdetr_r50vd | main |
| PekingU/rtdetr_r18vd | main |
| PekingU/rtdetr_r101vd | main |

### Example .env (traditional pipeline)

```bash
MODEL_ID=PekingU/rtdetr_r50vd
REVISION=main
VIDEO_PATH=./filter_example_video.mp4
```

---

## Open-vocabulary object detection (OWL-ViT)

**API:** `OwlViTProcessor` + `OwlViTForObjectDetection`. Models that load with this API are supported. Text queries at inference; no fixed class set.  
**Dependency:** `sentencepiece` (for OWL-ViT tokenizer).

**Example model IDs:**

| MODEL_ID | REVISION |
|----------|----------|
| google/owlvit-base-patch32 | main |

- **text_labels**: list of list of str (e.g. one list per image). Required for zero-shot.
- The script `zero_shot_object_detection_pipeline.py` uses a fixed model and `TEXT_LABELS` in code; only `VIDEO_PATH` (and optionally `THRESHOLD`, `PORT`) are read from .env.

### Example config (open-vocabulary, in code)

```python
FilterHuggingfaceVisionConfig(
    detection_type="open-vocabulary",
    model_id="google/owlvit-base-patch32",
    revision="main",
    text_labels=[["a person", "a cup"]],
    threshold=0.1,
)
```

### Example .env (zero-shot pipeline script)

```bash
VIDEO_PATH=./filter_example_video.mp4
THRESHOLD=0.1
PORT=8010
```

---

## Open-vocabulary object detection (Grounding DINO)

**API:** `AutoProcessor` + `AutoModelForZeroShotObjectDetection`. Any model on the Hub that loads with this API is supported (Grounding DINO, MM Grounding DINO). Open-vocabulary detection with text queries. **Dependency:** `timm` (for Swin backbone).

**Example model IDs:**

| MODEL_ID | REVISION |
|----------|----------|
| openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det | main |
| IDEA-Research/grounding-dino-tiny | main (if available) |

- **text_labels**: list of list of str (e.g. one list per image). Required.
- The script `scripts/grounding_dino.py` uses a fixed model and `TEXT_LABELS` in code; only `VIDEO_PATH` (and optionally `THRESHOLD`, `PORT`) are read from .env.

### Example config (open-vocabulary-grounding, in code)

```python
FilterHuggingfaceVisionConfig(
    detection_type="open-vocabulary-grounding",
    model_id="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
    revision="main",
    text_labels=[["a person", "a cup", "a cat"]],
    threshold=0.3,
)
```

### Example .env (Grounding DINO pipeline script)

```bash
VIDEO_PATH=./filter_example_video.mp4
THRESHOLD=0.3
PORT=8010
```
