#!/usr/bin/env python
"""Tests for open-vocabulary object detection task (Grounding DINO, no model download)."""

import unittest

from filter_huggingface_vision.backends.grounding_dino import (
    _normalize_results as _normalize_results_grounding,
)
from filter_huggingface_vision.backends.grounding_dino import _resolve_label
from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)
from filter_huggingface_vision.utils import as_bool


class TestNormalizeResultsGrounding(unittest.TestCase):
    """Unit tests for _normalize_results (grounding_dino backend)."""

    def test_empty_result_returns_empty_list(self):
        out = _normalize_results_grounding(
            {"boxes": None, "scores": []}, [], 100
        )
        self.assertEqual(out, [])

    def test_filters_scores_outside_zero_one(self):
        result = {
            "boxes": [[0, 0, 10, 10], [5, 5, 15, 15]],
            "scores": [0.7, -0.1],
        }
        out = _normalize_results_grounding(result, [["a", "b"]], 100)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["score"], 0.7)

    def test_output_schema(self):
        result = {
            "boxes": [[1, 2, 11, 12]],
            "scores": [0.95],
            "text_labels": ["cup"],
        }
        out = _normalize_results_grounding(result, [["cup"]], 100)
        self.assertEqual(len(out), 1)
        d = out[0]
        self.assertEqual(d["label"], "cup")
        self.assertEqual(d["score"], 0.95)
        self.assertEqual(d["box"]["format"], "xyxy")


class TestResolveLabel(unittest.TestCase):
    """Unit tests for _resolve_label (concatenated-span -> single configured phrase)."""

    def test_concatenated_synonyms_resolves_to_single_configured_phrase(self):
        phrases = ["a handgun", "a pistol", "a rifle"]
        resolved = _resolve_label("a handgun a pistol a rifle", phrases)
        # Never the concatenation; always a single configured phrase.
        self.assertIn(resolved, phrases)

    def test_longest_match_wins(self):
        phrases = ["gun", "a handgun"]
        # Both "gun" and "a handgun" are substrings; the longer/most specific wins.
        self.assertEqual(_resolve_label("a handgun", phrases), "a handgun")

    def test_tie_break_by_configured_order(self):
        # Equal length matches -> first in configured order wins.
        phrases = ["a cat", "a dog"]
        self.assertEqual(_resolve_label("a cat a dog", phrases), "a cat")

    def test_distinct_label_unchanged(self):
        phrases = ["a person", "a cup", "a cat"]
        self.assertEqual(_resolve_label("a person", phrases), "a person")

    def test_no_configured_phrase_returns_raw_unchanged(self):
        phrases = ["a handgun", "a pistol"]
        self.assertEqual(_resolve_label("a banana", phrases), "a banana")

    def test_case_and_whitespace_insensitive(self):
        phrases = ["a handgun", "a pistol"]
        self.assertEqual(_resolve_label("A Handgun ", phrases), "a handgun")

    def test_empty_phrases_returns_raw_unchanged(self):
        self.assertEqual(_resolve_label("a handgun a pistol", []), "a handgun a pistol")

    def test_substring_of_another_word_does_not_match(self):
        # Token-boundary match: "a cat" must not match inside "a caterpillar".
        self.assertEqual(
            _resolve_label("a caterpillar", ["a cat"]), "a caterpillar"
        )
        self.assertEqual(_resolve_label("category", ["cat"]), "category")

    def test_multi_word_phrase_matches_on_boundary(self):
        phrases = ["a cat", "a caterpillar"]
        self.assertEqual(_resolve_label("a cat a caterpillar", phrases), "a caterpillar")


class TestNormalizeResultsResolvesLabels(unittest.TestCase):
    """Label resolution in _normalize_results is opt-in (resolve_labels flag)."""

    def _concatenated_result(self):
        return {
            "boxes": [[0, 0, 10, 10], [5, 5, 15, 15]],
            "scores": [0.9, 0.8],
            # Model returns the concatenated union span for each box.
            "text_labels": [
                "a handgun a pistol a rifle",
                "a handgun a pistol a rifle",
            ],
        }

    def test_opted_in_every_emitted_label_is_a_configured_phrase(self):
        phrases = ["a handgun", "a pistol", "a rifle"]
        out = _normalize_results_grounding(
            self._concatenated_result(), [phrases], 100, resolve_labels=True
        )
        self.assertEqual(len(out), 2)
        for d in out:
            self.assertIn(d["label"], phrases)

    def test_default_off_preserves_concatenated_label(self):
        # Opt-in feature: without resolve_labels the raw concatenated label is kept.
        phrases = ["a handgun", "a pistol", "a rifle"]
        out = _normalize_results_grounding(self._concatenated_result(), [phrases], 100)
        self.assertEqual(len(out), 2)
        for d in out:
            self.assertEqual(d["label"], "a handgun a pistol a rifle")


class TestAsBool(unittest.TestCase):
    """as_bool coerces env-style config values without treating 'false' as truthy."""

    def test_truthy_values(self):
        for v in (True, 1, "1", "true", "True", "  TRUE  ", "yes", "on"):
            self.assertIs(as_bool(v), True, msg=repr(v))

    def test_falsy_values(self):
        for v in (False, 0, "0", "false", "False", "no", "", "garbage"):
            self.assertIs(as_bool(v), False, msg=repr(v))

    def test_none_uses_default(self):
        self.assertIs(as_bool(None), False)
        self.assertIs(as_bool(None, default=True), True)


class TestGroundingDinoConfig(unittest.TestCase):
    """Config validation for open-vocabulary-grounding (Grounding DINO) task."""

    def test_normalize_config_open_vocabulary_grounding_requires_text_labels(self):
        with self.assertRaises(ValueError) as ctx:
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
                    revision="main",
                    detection_type="open-vocabulary-grounding",
                )
            )
        self.assertIn("text_labels", str(ctx.exception).lower())

    def test_normalize_config_accepts_open_vocabulary_grounding_with_text_labels(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
                revision="main",
                detection_type="open-vocabulary-grounding",
                text_labels=[["a person", "a cup", "a cat"]],
            )
        )
        self.assertEqual(config.detection_type, "open-vocabulary-grounding")
        self.assertEqual(len(config.text_labels), 1)
        self.assertIn("person", config.text_labels[0][0])

    def _grounding_cfg(self, **overrides):
        base = dict(
            id="test",
            sources="",
            outputs="",
            model_id="openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det",
            revision="main",
            detection_type="open-vocabulary-grounding",
            text_labels=[["a person", "a cup"]],
        )
        base.update(overrides)
        return FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(**base)
        )

    def test_resolve_grounding_labels_survives_normalize_config(self):
        # Explicitly reconstructed, so the flag is preserved (not just via base merge).
        cfg = self._grounding_cfg(resolve_grounding_labels=True)
        self.assertIs(as_bool(getattr(cfg, "resolve_grounding_labels"), False), True)

    def test_resolve_grounding_labels_env_string_coerces(self):
        cfg = self._grounding_cfg(resolve_grounding_labels="true")
        self.assertIs(as_bool(getattr(cfg, "resolve_grounding_labels"), False), True)

    def test_resolve_grounding_labels_defaults_off(self):
        cfg = self._grounding_cfg()
        self.assertIs(as_bool(getattr(cfg, "resolve_grounding_labels"), False), False)


if __name__ == "__main__":
    unittest.main()
