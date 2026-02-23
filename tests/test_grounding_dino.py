#!/usr/bin/env python
"""Tests for open-vocabulary object detection task (Grounding DINO, no model download)."""

import unittest

from filter_huggingface_vision.backends.grounding_dino import (
    _normalize_results as _normalize_results_grounding,
)
from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)


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


if __name__ == "__main__":
    unittest.main()
