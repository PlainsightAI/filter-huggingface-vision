# Supported models by task

The filter supports multiple tasks via backends. Each task uses a different processor/model API. Output format is the same for all tasks: `frame.data["subjects"]["huggingface_vision"]` with `task`, `model`, `image`, and `detections` (label, score, box xyxy).

## Pipelines

| Script | Task | Model source |
|--------|------|--------------|
| `scripts/object_detection_pipeline.py` | object-detection | `MODEL_ID` + `REVISION` from .env |
| `scripts/zero_shot_object_detection_pipeline.py` | zero-shot-object-detection | Fixed in code: `google/owlvit-base-patch32` @ main |

The zero-shot script ignores `MODEL_ID`/`REVISION` in .env so the same .env (e.g. with `VIDEO_PATH`) can be shared without loading the wrong model.

---

## task: object-detection

Backend: `AutoImageProcessor` + `AutoModelForObjectDetection` (DETR, RT-DETR, Conditional DETR).  
**Dependency:** `timm` (for DETR/Conditional DETR backbones).

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

## task: zero-shot-object-detection

Backend: `OwlViTProcessor` + `OwlViTForObjectDetection` (open-vocabulary detection with text queries).  
**Dependency:** `sentencepiece` (for OWL-ViT tokenizer).

| MODEL_ID | REVISION |
|----------|----------|
| google/owlvit-base-patch32 | main |

- **text_labels**: list of list of str (e.g. one list per image). Required when using this task.
- The script `zero_shot_object_detection_pipeline.py` uses a fixed model and `TEXT_LABELS` in code; only `VIDEO_PATH` (and optionally `THRESHOLD`, `PORT`) are read from .env.

### Example config (zero-shot, in code)

```python
FilterHuggingfaceVisionConfig(
    task="zero-shot-object-detection",
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
