---
title: Class-name remapping
sidebar_label: Class-name remapping
sidebar_position: 3
---

Detection models output their own raw class names. Open-vocabulary models
(OWL-ViT, Grounding DINO) return the prompt itself — e.g. `"a handgun"`,
`"a shotgun"` — and closed-vocabulary models (RT-DETR, DETR) return their own
labels — e.g. `"person"`, `"car"`. This feature lets you choose the **final
name** that appears in `meta.detections[].class` and in the visualization
overlay.

All options are optional and backward compatible: with none set, raw labels pass
through unchanged.

## Options

| Config | Applies to | Purpose |
|--------|-----------|---------|
| `text_labels` (inline string) | open-vocabulary | Rename per prompt using `finalName\|\|\|prompt` items joined by `###`. |
| `class_delimiter` | open-vocabulary | Separator between `finalName` and `prompt` (default `\|\|\|`). |
| `prompt_delimiter` | open-vocabulary | Separator between items (default `###`). |
| `label_map` | any model | Explicit `{raw: final}` rename. |
| `collapse_labels_to` | any model | Force every detection to this single name. |

**Precedence:** `collapse_labels_to` > `label_map` > raw label.

## Inline mapping (open-vocabulary)

```
text_labels = "gun|||a handgun###gun|||a shotgun"
                └┬┘ └────┬────┘
              final    prompt sent to the model
```

The model still receives `a handgun` / `a shotgun`; both detections are reported
as `gun`. An item with no `|||` is a bare prompt that maps to itself. The
existing `list[list[str]]` form (e.g. `[["a person", "a cup"]]`) keeps working
with no rename.

## Explicit map (any model, incl. closed-vocabulary)

```python
label_map = {"person": "people", "car": "vehicle"}
```

## Collapse everything

```python
collapse_labels_to = "weapon"   # every detection's class becomes "weapon"
```

## Example

`scripts/weapon_label_remap.py` runs **VideoIn → FilterHuggingfaceVision → Webvis**
on a weapon-detection video, reporting all weapon prompts as `gun`:

```env
VIDEO_PATH=./your-video.mp4
TEXT_LABELS=gun|||a handgun###gun|||a shotgun###gun|||a pistol###gun|||a rifle
# Or collapse everything instead:
# COLLAPSE_LABELS_TO=weapon
```

```bash
python scripts/weapon_label_remap.py
```

## Grounding DINO note

`label_map` and inline `text_labels` mapping match the detected label against
the map **by exact key**. OWL-ViT emits the prompt verbatim (`"a handgun"`), so
the rename always applies. Grounding DINO (`open-vocabulary-grounding`) can emit
a sub-phrase or token concatenation instead of the full prompt (e.g. `"handgun"`
for the prompt `"a handgun"`), so an exact-key lookup may not match and the
rename silently no-ops. For Grounding DINO, prefer **`collapse_labels_to`** (it
ignores keys and always applies), or key your `label_map` on the labels the model
actually emits.

## Scope

Applies to object detection only (`open-vocabulary`,
`open-vocabulary-grounding`, `closed-vocabulary`). `image-classification` and
`embedding` outputs are left unchanged.
