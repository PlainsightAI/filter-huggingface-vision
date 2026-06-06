# Design: clean per-detection labels for Grounding DINO (PLAT-1105)

## Problem

With `detection_type=open-vocabulary-grounding`, detected class names come back
concatenated/garbled — e.g. `"a handgun a pistol a rifle"` for every box, or
`"a a pistol a handgun"`.

## Root cause (verified with the real model)

`IDEA-Research/grounding-dino-tiny` via `post_process_grounded_object_detection`
returns, for each box, a `text_labels` string that is the **union of every input
phrase whose tokens matched that box**. When the prompts are semantically
overlapping synonyms (`"a handgun"`, `"a pistol"`, `"a rifle"`), the gun region
matches all three, so the label becomes `"a handgun a pistol a rifle"`. This is
independent of how text is fed (list-of-list vs period-separated string — both
reproduce it). With distinct prompts (`"a person"`, `"a cup"`) the labels are
already clean.

So the fix is **not** a change to the processor input format; it is resolving the
concatenated span back to a single configured phrase per box.

## Approach

**Opt-in, default off.** Mirroring how PLAT-1104 keeps remapping user-controlled,
this resolution is *not* applied automatically. The model's verbatim output is
preserved unless the user sets a new config flag:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `resolve_grounding_labels` | `bool` | `false` | When true, collapse each box's concatenated label to a single configured phrase. |

Rationale: the concatenated span is the model's actual output, and the
collapse-to-single-phrase step is a heuristic (longest/most-specific match) that can
pick the "wrong" synonym. Forcing it on every Grounding DINO user would silently
transform data, so it stays explicit — the user turns it on when the cleaner label
is what they want. Booleans arriving from env vars as strings (`"false"`) are coerced
via a shared `as_bool` helper so they aren't treated as truthy.

When enabled, in `grounding_dino.py:_normalize_results`, after obtaining the raw
label string for a box, resolve it to a single clean label using the configured
phrases (`text_labels` for this image):

1. Build the candidate phrase list (the prompts the model was given).
2. For the raw label string, find which configured phrases are contained in it
   (case-insensitive, whitespace-normalized substring match).
3. Choose the **longest** matching phrase (most specific); tie-break by the order
   the phrases were configured.
4. If no configured phrase matches (e.g. model returned an unrelated span), keep
   the raw label unchanged (no silent data loss).

This is deterministic, needs no extra forward passes, and yields one configured
phrase per box. It composes with PLAT-1104 remapping: a clean single label means
`label_map` / inline mapping match reliably on the grounding backend.

`_normalize_results` takes the configured phrase list (already passed as
`text_labels_list`) plus a `resolve_labels` flag, and applies the resolution per
detection only when the flag is set.

## Out of scope

- OWL-ViT (`open-vocabulary`) — already returns the verbatim phrase; untouched.
- Per-phrase confidence attribution / re-scoring. We pick by specificity + order,
  not by re-running the model per phrase.

## Testing

`tests/test_grounding_dino.py` (extend) + a focused unit test of the resolver:

- raw label `"a handgun a pistol a rifle"` with phrases
  `["a handgun","a pistol","a rifle"]` and `resolve_labels=True` → resolves to a
  single configured phrase (longest/first); never the concatenation.
- **default off:** same input with `resolve_labels` unset → concatenated label
  preserved verbatim.
- distinct-phrase label `"a person"` → unchanged.
- raw label containing no configured phrase → unchanged (fallback).
- case/whitespace-insensitive match (`"A Handgun"` → `"a handgun"`).
- `as_bool` coercion of env-style string flags (`"true"`/`"false"`/`"1"`/`""` …).
- assert every emitted label is one of the input phrases (the ticket's
  acceptance criterion) for the overlapping-synonyms case when opted in.

Pure resolver function is unit-tested in isolation; one end-to-end test drives the
real model on a frame to confirm no concatenated labels are emitted when enabled.
