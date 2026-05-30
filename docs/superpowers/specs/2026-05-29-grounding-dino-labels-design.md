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

In `grounding_dino.py:_normalize_results`, after obtaining the raw label string for
a box, resolve it to a single clean label using the configured phrases
(`text_labels` for this image):

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

`_normalize_results` will take the configured phrase list (already passed as
`text_labels_list`) and apply the resolution per detection.

## Out of scope

- OWL-ViT (`open-vocabulary`) — already returns the verbatim phrase; untouched.
- Per-phrase confidence attribution / re-scoring. We pick by specificity + order,
  not by re-running the model per phrase.

## Testing

`tests/test_grounding_dino.py` (extend) + a focused unit test of the resolver:

- raw label `"a handgun a pistol a rifle"` with phrases
  `["a handgun","a pistol","a rifle"]` → resolves to a single configured phrase
  (longest/first); never the concatenation.
- distinct-phrase label `"a person"` → unchanged.
- raw label containing no configured phrase → unchanged (fallback).
- case/whitespace-insensitive match (`"A Handgun"` → `"a handgun"`).
- assert every emitted label is one of the input phrases (the ticket's
  acceptance criterion) for the overlapping-synonyms case.

Pure resolver function is unit-tested in isolation; one end-to-end test drives the
real model on a frame to confirm no concatenated labels are emitted.
