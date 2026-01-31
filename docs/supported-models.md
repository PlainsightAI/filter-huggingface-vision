# Supported models by detection type

The filter supports two detection variants via backends; each uses a different processor/model API. Output format is the same: `frame.data["subjects"]["huggingface_vision"]` with `detection_type`, `model`, `image`, and `detections` (label, score, box xyxy). Config uses `detection_type`: `"closed-vocabulary"` or `"open-vocabulary"`.

## Pipelines

| Script | Detection type | Model source |
|--------|----------------|--------------|
| `scripts/object_detection_pipeline.py` | Closed-vocabulary (DETR / RT-DETR) | `MODEL_ID` + `REVISION` from .env |
| `scripts/zero_shot_object_detection_pipeline.py` | Open-vocabulary (OWL-ViT) | Fixed in code: `google/owlvit-base-patch32` @ main |

The zero-shot script ignores `MODEL_ID`/`REVISION` in .env so the same .env (e.g. with `VIDEO_PATH`) can be shared without loading the wrong model.

---

## Closed-vocabulary object detection (DETR / RT-DETR)

Fixed set of classes (e.g. COCO). Backend: `AutoImageProcessor` + `AutoModelForObjectDetection` (DETR, RT-DETR, Conditional DETR).  
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

## Open-vocabulary object detection (OWL-ViT)

Text queries at inference; no fixed class set. Backend: `OwlViTProcessor` + `OwlViTForObjectDetection`.  
**Dependency:** `sentencepiece` (for OWL-ViT tokenizer).

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
