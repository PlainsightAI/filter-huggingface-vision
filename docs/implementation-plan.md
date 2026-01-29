---
title: Implementation Plan (Object Detection MVP)
sidebar_label: Implementation Plan
sidebar_position: 2
---

# filter-huggingface-vision: Object Detection MVP (Generic HF)

## Central premise

**`filter-huggingface-vision` can download and run "any" HF object detection model** using the Transformers pattern: `AutoImageProcessor + AutoModelForObjectDetection + post_process_object_detection`. RT-DETR is **one of the smoke-test models**, not the exclusive path. The architecture leaves a **hook for classification and other tasks** later; **MVP is object detection only**.

---

## MVP objective

Deliver one OpenFilter filter: `filter-huggingface-vision`, capable of:

- Downloading a model from Hugging Face by `model_id` + `revision` (required)
- Running **object detection** via:
  - `AutoImageProcessor.from_pretrained()`
  - `AutoModelForObjectDetection.from_pretrained()`
  - `image_processor.post_process_object_detection(...)`
- Producing **output in a stable schema** (OpenFilter subject data)
- Running smoke tests in CI with at least **RT-DETR + 1–2 additional models**

---

## 1) Filter contract (OpenFilter)

### Input

- A `Frame` with image (RGB via PIL / numpy as OpenFilter exposes).
- Filter config defines what to run.

### Output (subject data)

Attach results at: `frame.data["subjects"]["huggingface_vision"]`.

**Schema (object detection):**

```json
{
  "task": "object-detection",
  "model": { "id": "...", "revision": "..." },
  "image": { "width": 0, "height": 0 },
  "detections": [
    {
      "label": "string",
      "score": 0.0,
      "box": { "format": "xyxy", "xmin": 0.0, "ymin": 0.0, "xmax": 0.0, "ymax": 0.0 }
    }
  ]
}
```

---

## 2) Config (MVP)

**Fields:** `model_id` (required), `revision` (required), `task` (default `"object-detection"`), `threshold` (default `0.3`), `device` (default CPU), `trust_remote_code` (default `false`), `max_detections` (e.g. 100), `input_topic` / `output_topic` (default `"main"`).

**`normalize_config()`:** Reject empty `revision`; validate `threshold ∈ [0, 1]`; validate `task` (MVP: only `object-detection`); enforce `trust_remote_code=false` or allowlist.

---

## 3) Generic object-detection runtime (core)

### Setup

- Pinned download with `model_id` + `revision`; HF cache.
- Load: `AutoImageProcessor.from_pretrained(...)`, `AutoModelForObjectDetection.from_pretrained(...).to(device)`, `model.eval()`.

### Process

- Get image and size from frame.
- Preprocess with `image_processor`, run `model(**inputs.to(device))` under `torch.no_grad()`.
- `post_process_object_detection(outputs, threshold=..., target_sizes=...)`.
- Normalize to schema: map labels via `id2label`, sort by score, apply `max_detections`, output xyxy boxes.

---

## 4) “Any model” vs multiple methods

- **Method A (default, MVP):** `AutoImageProcessor` + `AutoModelForObjectDetection` + `post_process_object_detection`. If load fails, raise a clear “model not compatible” error.
- **Method B (Phase 2):** Optional `pipeline("object-detection")` fallback.
- **Method C (future):** Specific adapters (GroundingDINO, OWL-ViT, YOLOX-S, etc.).

---

## 5) Smoke tests (CI) — RT-DETR + 2 more

**Models:** e.g. `PekingU/rtdetr_r50vd`, `facebook/detr-resnet-50`, `microsoft/conditional-detr-resnet-50`, each with **pinned revision**.

**Assert:** Schema present; `detections` list; scores in [0,1]; valid boxes (xmin&lt;xmax, ymin&lt;ymax); `model.id` / `model.revision` match config. Use a small fixed image in repo; cache HF in CI when possible.

---

## 6) Deliverables by phase

- **Phase 1 — MVP:** Generic Auto* object detection, stable schema, smoke tests with 3 models, `revision` required.
- **Phase 2:** Optional pipeline fallback, clearer errors, optional allowlist.
- **Phase 3:** Multi-task (e.g. `image-classification` with `AutoModelForImageClassification` + `top_k`).

---

## 7) Governance

- `trust_remote_code = false` by default; require `revision`; optional allowlist; log `model_id`, `revision`, device, transformers/torch versions.

---

## Expected outcome

Point the filter at any compatible HF object-detection repo (e.g. fine-tuned DETR): set `model_id` and `revision`, filter downloads and runs, returns detections in the standard schema. RT-DETR is one of the smoke tests proving the generic path works.
