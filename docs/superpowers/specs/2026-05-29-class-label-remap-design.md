# Design: configurable class-name remapping (PLAT-1104)

## Problem

Detection backends emit the model's raw class name. For open-vocabulary models
(OWL-ViT, Grounding DINO) the label is the prompt itself — e.g. `"a handgun"`,
`"a shotgun"`. For closed-vocabulary models (rtdetr etc.) it is the model's own
label — e.g. `"person"`, `"car"`.

Today that raw label flows untouched into `meta.detections[].class` and into the
drawn visualization. The filter user cannot choose a cleaner final name, cannot
rename `"a handgun"` → `"gun"`, and cannot collapse several labels into one
(e.g. everything → `"weapon"`).

## Goal

Let the filter user choose the final class name that appears in the output and
the visualization. It must be easy to configure (env-friendly), and consistent
with how `PlainsightAI/filter-sam3-detector` already does this (`prompt_label_map`).

## Scope

- **In scope:** object detection — both `open-vocabulary` / `open-vocabulary-grounding`
  and `closed-vocabulary`.
- **Out of scope:** `image-classification` and `embedding` detection types. Their
  label fields are left untouched (documented, asserted by tests).

## Configuration

New / extended fields on `FilterHuggingfaceVisionConfig`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `text_labels` | `str \| list[list[str]] \| None` | `None` | **Extended.** Now also accepts an inline-mapping string (sam3 syntax). The existing `list[list[str]]` form keeps working unchanged (no rename). |
| `class_delimiter` | `str` | `"\|\|\|"` | Separates `finalName` from `prompt` within one item. |
| `prompt_delimiter` | `str` | `"###"` | Separates items from each other. |
| `label_map` | `dict[str, str] \| None` | `None` | Explicit `raw → final` rename. Works for **any** detection type, including closed-vocabulary. |
| `collapse_labels_to` | `str \| None` | `None` | Easy mode: every detection's class becomes this single value. |

### Inline syntax (open-vocabulary)

```
text_labels = "gun|||a handgun###gun|||a shotgun"
                └┬┘ └────┬────┘
              final    prompt sent to the model
```

- Items split on `prompt_delimiter` (`###`).
- Within an item, split on `class_delimiter` (`|||`) as `finalName|||prompt`.
- An item with no `class_delimiter` is a bare prompt that maps to itself
  (`"truck"` → `"truck"`).
- Parsing produces:
  - `text_labels` normalized to `list[list[str]]` of the **prompts** the model
    receives — e.g. `[["a handgun", "a shotgun"]]`.
  - entries merged into the effective label map — e.g.
    `{"a handgun": "gun", "a shotgun": "gun"}`.

### Precedence

When resolving the final name for a detected raw label:

```
collapse_labels_to  >  label_map (explicit ∪ inline-derived)  >  raw label
```

`collapse_labels_to`, when set, ignores keys entirely and overrides all labels.

## Architecture / data flow

```
backend.run(image) -> [{label, score, box}, ...]
        │
        ▼
process(): payload["detections"]
        │
        ▼
_apply_label_map(detections, label_map, collapse_labels_to)   # mutate label in place, once
        │
        ├──> _payload_to_meta_format  -> meta.detections[].class   (final name)
        └──> _create_visualization    -> overlay text              (final name)
```

The remap runs **once** on `payload["detections"]` in `process()`, right after
`backend.run()`. Because both `_payload_to_meta_format` and `_create_visualization`
read the same payload, meta and viz are guaranteed to agree. This also cleans up
the overlapping/ugly raw labels seen in the bug report screenshot.

### Components

- **`_parse_text_labels(value, class_delimiter, prompt_delimiter)`** (new, in
  `filter.py` or `utils.py`): pure function. Input the raw `text_labels` value +
  delimiters; output `(prompts_list_of_list, inline_label_map)`. Handles string,
  list, and `None`. Pure and unit-testable in isolation.
- **`_apply_label_map(detections, label_map, collapse_labels_to)`** (new): pure
  function mutating/returning detection dicts with the resolved final `label`.
- **`normalize_config`**: calls `_parse_text_labels`, stores normalized
  `text_labels` + effective `label_map` (inline ∪ explicit; explicit wins on key
  conflict), validates delimiters and duplicates.
- **`process`**: calls `_apply_label_map` on the detection payload before
  building meta / viz.

## Validation (mirrors sam3)

`normalize_config` raises `ValueError` when:
- `class_delimiter` or `prompt_delimiter` is empty.
- `class_delimiter == prompt_delimiter`.
- the same prompt is mapped to two different final names (duplicate prompt
  mapping).
- `text_labels` is neither `str`, `list`, nor `None`.
- `label_map` is set but is not a `dict[str, str]`.
- `collapse_labels_to` is set but is not a non-empty `str`.

Existing validation (open-vocabulary requires non-empty `text_labels`, etc.) is
preserved.

## Backward compatibility

- `text_labels=[["a person", "a cup"]]` (current scripts) → unchanged: prompts
  pass through, label map empty, no rename.
- No new field is required; all default to "no remapping".

## Testing

New file `tests/test_label_remap.py` (plus targeted additions to
`tests/test_filter_config.py` if config-shape assertions fit better there).

**Parsing — `_parse_text_labels` (mirror sam3's cases):**
- `None` → `(None, {})`.
- bare prompts `"car###truck###dog"` → prompts `[["car","truck","dog"]]`, map
  `{"car":"car","truck":"truck","dog":"dog"}` (each maps to itself, matching sam3).
- inline mapping `"vehicle|||car###vehicle|||truck###animal|||dog"` →
  prompts `[["car","truck","dog"]]`, map `{"car":"vehicle","truck":"vehicle","dog":"animal"}`.
- mixed mapped + bare items.
- whitespace trimming around delimiters.
- custom delimiters (`class_delimiter="="`, `prompt_delimiter="|"`).
- existing `list[list[str]]` input → prompts unchanged, map `{}`.
- invalid type (e.g. dict) → `ValueError`.

**Validation:**
- empty `class_delimiter` / empty `prompt_delimiter` → `ValueError`.
- equal delimiters → `ValueError`.
- duplicate prompt mapping (`"vehicle|||car###automobile|||car"`) → `ValueError`.

**Remap — `_apply_label_map`:**
- `label_map` renames a closed-vocabulary label (`"person"` → `"people"`).
- `collapse_labels_to="weapon"` turns every detection into `"weapon"`.
- precedence: `collapse_labels_to` beats `label_map` beats raw.
- unmapped label passes through untouched.

**End-to-end (filter level):**
- the gun scenario — open-vocab payload with raw labels `"a handgun"`/`"a shotgun"`
  and `text_labels="gun|||a handgun###gun|||a shotgun"` → both
  `meta.detections[].class` are `"gun"`.
- meta and visualization show the **same** final name (assert overlay text uses
  remapped label).
- out-of-scope types: a `classification` payload is left untouched by the remap.

Follow the existing test style in `tests/` and keep tests TDD-first (write the
failing test, then implement).
