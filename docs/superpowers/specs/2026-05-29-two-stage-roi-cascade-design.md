# Design: two-stage ROI-gated detection (PLAT-1106)

## Problem

Small/occluded targets (e.g. a handgun in CCTV) are missed on the full frame but
detected reliably when zoomed into a region of interest. Person→crop→detector
lifts confidence substantially (~0.2 → ~0.7). Doing this today needs an external
filter-hf → filter-crop → filter-hf chain, and the final boxes stay in crop
coordinates.

## Goal

Optional in-filter cascade: a **gate** detector finds regions, the frame is
cropped to each (padded) gate box, the **main** detector runs on each crop, and
the resulting boxes are remapped to full-frame coordinates — emitting the normal
`meta.detections` + visualization on the full frame. With no gate config, behavior
is exactly as today (single stage).

## Configuration (all optional; cascade active only when `gate_model_id` is set)

| Field | Default | Purpose |
|-------|---------|---------|
| `gate_model_id` | `None` | Model id for the gate detector. Unset → single-stage. |
| `gate_detection_type` | `"closed-vocabulary"` | Gate task (e.g. closed-vocab person detector, or open-vocab). |
| `gate_revision` | `"main"` | Gate model revision. |
| `gate_prompt` | `None` | Gate text labels (required when gate is open-vocabulary). |
| `gate_threshold` | `= threshold` | Gate confidence threshold. |
| `gate_class` | `None` | Keep only gate detections whose class equals this (e.g. `"person"`). `None` → keep all. |
| `gate_pad` | `0.3` | Fractional padding added to each side of a gate box before cropping. |
| `gate_max_regions` | `5` | Cap on crops per frame (cost guard; logged when exceeded). |

## Architecture / data flow

```
full image
   │
   ├─ gate_backend.run(full)  ──► gate detections (pixel xyxy)
   │        │ filter by gate_class, pad by gate_pad, clamp to frame, cap at gate_max_regions
   │        ▼
   │   [ (x0,y0,x1,y1), ... ]  crop regions
   │
   └─ for each region:
        crop = image[y0:y1, x0:x1]
        dets = main_backend.run(crop)         # pixel xyxy in CROP coords
        remap: x += x0, y += y0               # → FULL-frame pixel coords
   ▼
aggregate dets → existing payload path (_apply_label_map, _payload_to_meta_format, viz)
```

- A second backend instance (`self._gate_backend`) is loaded in `setup()` only when
  `gate_model_id` is set; `shutdown()` releases it.
- The gate uses the same backend registry/run contract as the main detector.
- Remapping is the pure, testable core: crop-local pixel box + crop origin →
  full-frame pixel box. `_payload_to_meta_format` then normalizes by full W/H as
  today, so downstream schema is unchanged.
- All remapped detections across regions are concatenated into one payload, so
  meta + viz render on the full frame. Existing PLAT-1104 remapping still applies.

## Decisions / scope

- **Coordinate space:** gate and main backends already return pixel xyxy when
  `target_sizes` is passed; remap is integer translation by crop origin.
- **Padding:** `gate_pad` is a fraction of the gate box width/height, added on each
  side, then clamped to `[0,W]×[0,H]`.
- **Duplicate targets** (same object inside two overlapping gate crops): out of
  scope for v1 — emit all; a follow-up can add IoU dedup/NMS. Logged if regions
  overlap heavily. (Keeps v1 focused.)
- **Single-stage unchanged:** when `gate_model_id` is unset, no gate backend is
  loaded and `process()` takes the existing path verbatim.
- **Cost:** N regions ⇒ N main-detector forward passes; `gate_max_regions` bounds
  it and is logged when regions are dropped (no silent cap).

## Testing

- **Remap math (pure, unit):** crop origin `(x0,y0)` + crop-local box
  `(cx0,cy0,cx1,cy1)` → full-frame `(x0+cx0, …)`; normalized rois land in the right
  place on a known frame size.
- **Gate filtering/padding (pure, unit):** `gate_class` filter keeps only matching
  boxes; `gate_pad=0.3` expands and clamps correctly at frame edges;
  `gate_max_regions` caps and logs.
- **Single-stage passthrough:** no gate config → identical payload to today
  (regression guard).
- **End-to-end (real model):** person-gated OWL-ViT on the weapon clip boxes the
  handgun on the full frame at the correct location, confidence higher than the
  full-frame single stage.

TDD: pure helpers (`_pad_and_clamp_region`, `_remap_box_to_full`) first, then wire
into `process()`.
