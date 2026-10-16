# Supported models by detection type

The filter supports a fixed set of **Hugging Face APIs**. Each API is identified by a processor + model class pair and corresponds to one `detection_type`. **Each API supports all models on the Hugging Face Hub that are compatible with that API**—any model that can be loaded with the same processor and model classes will work. Below we list each supported API and **example model IDs** (tested or commonly used); the Hub may list additional compatible models for each API.

## Supported APIs (summary)

| HF API (processor + model) | `detection_type` | Example model IDs |
|----------------------------|------------------|-------------------|
| `AutoImageProcessor` + `AutoModelForImageClassification` | `image-classification` | `google/vit-base-patch16-224`, `facebook/convnext-tiny-224` |
| `AutoImageProcessor` + `AutoModelForObjectDetection` | `closed-vocabulary` | `PekingU/rtdetr_r50vd`, `facebook/detr-resnet-50` |
| `AutoProcessor` + `AutoModelForZeroShotObjectDetection` | `open-vocabulary` | `google/owlv2-base-patch16-ensemble`, `google/owlvit-base-patch32` |
| `AutoProcessor` + `AutoModelForZeroShotObjectDetection` | `open-vocabulary-grounding` | `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det` |
| `AutoModel` / any `AutoModelFor*` / timm (hook-based) | `embedding` | `facebook/dinov2-small`, `facebook/dinov2-base`, `google/vit-base-patch16-224`, `convnext_tiny.dinov3_lvd1689m` (timm) |

**Output format:** All results are written to `frame.data["meta"]`. Upstream meta (`id`, `ts`, `src`, `src_fps`) is preserved.

- **Object detection:** `detections`, `detection_confidence`, `detection_type`, `task`, `model`.
- **Image classification:** no `detections` nor `detection_confidence`. Only `classification`: `{ "classes", "confidences", "architecture", "timestamp", "filter_id", "model_id", "revision", "top_k" }`, plus `detection_type`, `task`, `model`.
- **Embedding:** `embedding` (feature vector) and optionally `min_exemplar_distance` written to `frame.data`. Metadata (`detection_type`, `task`, `model`) in `frame.data["meta"]`.
- **Per-detection embeddings:** `meta.detections[].embeddings` when a detector sets `embed_scope` to `detections` or `both`. See [Per-detection embeddings](#per-detection-embeddings).

## Pipelines

| Script | Detection type | Model source |
|--------|----------------|--------------|
| `scripts/object_detection.py` | Closed-vocabulary (DETR / RT-DETR) | `MODEL_ID` + `REVISION` from .env |
| `scripts/image_classification.py` | Image classification (ViT / ConvNeXt) | `MODEL_ID` + `REVISION` from .env |
| `scripts/zero_shot_object_detection.py` | Open-vocabulary (OWLv2 / OWL-ViT) | Fixed in code: `google/owlv2-base-patch16-ensemble` @ main |
| `scripts/grounding_dino.py` | Open-vocabulary (Grounding DINO) | Fixed in code: MM Grounding DINO tiny @ main |
| `scripts/generate_exemplars.py` | Embedding (offline exemplar generation) | `MODEL_ID` + `REVISION` from .env |

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

### Output (image-classification)

`frame.data["meta"]` has no `detections` nor `detection_confidence`. It includes `classification` (Protege-like) and method info:

```json
{
  "id": 38,
  "ts": 1761090922.42,
  "src": "file:///path/to/video.mp4",
  "src_fps": 25.0,
  "detection_type": "image-classification",
  "task": "image-classification",
  "model": { "id": "facebook/convnext-tiny-224", "revision": "main" },
  "classification": {
    "classes": ["tabby cat", "Egyptian cat"],
    "confidences": [0.42, 0.31],
    "architecture": "huggingface",
    "timestamp": 1761090922.42,
    "filter_id": "filter_huggingface_vision",
    "model_id": "facebook/convnext-tiny-224",
    "revision": "main",
    "top_k": 5
  }
}
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

## Open-vocabulary object detection (OWLv2 / OWL-ViT)

**API:** `AutoProcessor` + `AutoModelForZeroShotObjectDetection`. Both OWLv2 (`google/owlv2-*`) and OWLv1 (`google/owlvit-*`) models are supported — the Auto classes select the correct architecture from the checkpoint `config.json`. Text queries at inference; no fixed class set.  
**Dependency:** `sentencepiece` (for the tokenizer).

**Example model IDs:**

| MODEL_ID | REVISION |
|----------|----------|
| google/owlv2-base-patch16-ensemble | main |
| google/owlv2-base-patch16 | main |
| google/owlvit-base-patch32 | main |
| google/owlvit-base-patch16 | main |

- **text_labels**: list of list of str (e.g. one list per image). Required for zero-shot.
- The script `scripts/zero_shot_object_detection.py` uses a fixed model and `TEXT_LABELS` in code; only `VIDEO_PATH` (and optionally `THRESHOLD`, `PORT`) are read from .env.
- For best throughput on GPU, use `torch_dtype=torch.float16` (enabled by default in the backend) and pre-resize input video to 960×540 (`!resize=960x540` in the VideoIn source string).

### Example config (open-vocabulary, in code)

```python
FilterHuggingfaceVisionConfig(
    detection_type="open-vocabulary",
    model_id="google/owlv2-base-patch16-ensemble",  # or "google/owlvit-base-patch32" for OWLv1
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

---

## Per-detection embeddings

A detector (`closed-vocabulary`, `open-vocabulary`, or `open-vocabulary-grounding`) can publish an L2-normalized vector for selected boxes in the same pass as the detections. This is not `detection_type=embedding`. The whole-frame embedding path below is unchanged.

Set `embed_scope` to `detections` (crops only), `frame` (one vector for the frame), or `both`. Crop vectors are written on `meta.detections[].embeddings` as `{"model": "<repo>", "vector": [...]}`. A detection that fails the gate stays in the list with no `embeddings` key.

`embedding_model_id` is the crop model. `appearance_model_id`, when set, adds a second vector from the unaligned crop. Each model that is not the detector needs its own revision (`embedding_revision`, `appearance_revision`). The detector SHA is reused only when the embedder is that same repo.

`embed_classes` names the labels that receive a vector. `""` and `[]` are rejected. Gates, all optional: `embed_min_score` (default 0.5), `embed_min_bbox_area` (pixels squared, default 1024; a fraction in (0, 1) is rejected; `0` disables), `embed_reject_border` (default true), `embed_min_sharpness` (Laplacian variance, default 20; `0` disables). `embed_max_detections` (`0` means no cap) keeps the highest-scoring crops. Passing crops share one forward per model.

`embed_align_crops` warps each selected face onto the ArcFace 112 square from five RetinaFace landmarks and loads the identity weights with a local IResNet (`ir18` through `ir200`, including `minchul/cvlface_arcface_ir101_webface4m`). Hub code is not executed. A face with no matched landmarks, or a crop that is not square, is left without vectors. An identity repo has no `preprocessor_config.json`; pointing `embedding_model_id` at one without `embed_align_crops` fails at setup and names that flag. `landmark_revision` pins the RetinaFace weights and defaults to a commit SHA. `embed_scope` of `frame` or `both` together with `embed_align_crops` requires `appearance_model_id`, so the identity repo is not loaded again as a whole-frame embedder.

### Example config (face identity + appearance)

```python
FilterHuggingfaceVisionConfig(
    detection_type="open-vocabulary-grounding",
    model_id="IDEA-Research/grounding-dino-tiny",
    revision="a2bb814dd30d776dcf7e30523b00659f4f141c71",
    text_labels=[["a face"]],
    embed_scope="detections",
    embed_classes=["face"],
    embed_align_crops=True,
    embedding_model_id="minchul/cvlface_arcface_ir101_webface4m",
    embedding_revision="5846a5e96fbe1094952a414c78ce5b057c92d013",
    appearance_model_id="facebook/dinov2-small",
    appearance_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
)
```

The identity vector is 512-d. The appearance vector is the DINOv2 width (384 for `dinov2-small`). Both lists are L2-normalized. `docker-compose.face-embeddings.yaml` is the same pipeline.

---

## Embedding extraction (any vision model)

**API:** Model-agnostic — uses PyTorch forward hooks to extract penultimate-layer features from any vision model. Works with:

- **HuggingFace Transformers** (`model_loader="transformers"`): Tries `AutoModel` first (headless). If that fails, falls back to headed classes (`AutoModelForImageClassification`, `AutoModelForObjectDetection`, etc.) and hooks the last layer before the output head. This means you can extract embeddings from *any* HF vision model — classification, detection, segmentation, depth estimation — without code changes.
- **timm** (`model_loader="timm"`): Uses `timm.create_model(name, num_classes=0)` which natively strips the classifier head.

**Example model IDs:**

| MODEL_ID | REVISION | Notes |
|----------|----------|-------|
| facebook/dinov2-small | main | Pure feature extractor (384-d, loaded via HF Transformers) |
| facebook/dinov2-base | main | Pure feature extractor (768-d, loaded via HF Transformers) |
| google/vit-base-patch16-224 | main | Classification model — hook extracts pre-head features |
| facebook/detr-resnet-50 | main | Detection model — hook extracts backbone features |
| convnext_tiny.dinov3_lvd1689m | main | DINOv3-distilled ConvNeXt (requires `model_loader="timm"`) |
| convnext_small.dinov3_lvd1689m | main | DINOv3-distilled ConvNeXt, larger (requires `model_loader="timm"`) |

### Example config (embedding, in code)

```python
FilterHuggingfaceVisionConfig(
    detection_type="embedding",
    model_id="facebook/dinov2-small",  # HF model (default); set model_loader="timm" for timm models
    revision="main",
    exemplar_embeddings_path="./exemplars.npz",  # optional
    output_embeddings=True,
    output_distances=True,
)
```

### Example .env (embedding pipeline)

```bash
MODEL_ID=facebook/dinov2-small
REVISION=main
VIDEO_PATH=./filter_example_video.mp4
EXEMPLAR_EMBEDDINGS_PATH=./exemplars.npz
```

### Generating exemplar embeddings

Use `scripts/generate_exemplars.py` to extract embeddings from a directory of reference images:

```bash
MODEL_ID=facebook/dinov2-small
REVISION=main
IMAGE_DIR=./reference_images
OUTPUT_PATH=./exemplars.npz

python scripts/generate_exemplars.py
```

The `.npz` file contains an `embeddings` key with shape `(N, D)` where N is the number of reference images and D is the embedding dimensionality.

### Output (embedding)

Embedding output is written to `frame.data` (not nested under `meta`):

```json
{
  "meta": {
    "detection_type": "embedding",
    "task": "embedding",
    "model": { "id": "facebook/dinov2-small", "revision": "main" }
  },
  "embedding": [0.0123, -0.0456, 0.0789, "..."],
  "min_exemplar_distance": 0.42
}
```

- `embedding`: L2-normalized feature vector. Dimensionality depends on the model (e.g. 384 for DINOv2-small, 768 for DINOv2-base/ViT-base).
- `min_exemplar_distance`: Only present when `exemplar_embeddings_path` is set. L2 distance from the raw model vector to the closest exemplar — lower values mean the frame is more similar to the reference set. The published `embedding` is L2-normalized after that distance is computed, so the two numbers do not live in the same space.
