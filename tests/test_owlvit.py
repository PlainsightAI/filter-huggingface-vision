#!/usr/bin/env python
"""Tests for open-vocabulary object detection task (OWL-ViT, no model download)."""

import unittest

from filter_huggingface_vision.backends.owlvit import (
    _normalize_results as _normalize_results_owlvit,
)
from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)


class TestNormalizeResultsOwlvit(unittest.TestCase):
    """Unit tests for _normalize_results (owlvit backend)."""

    def test_empty_result_returns_empty_list(self):
        out = _normalize_results_owlvit(
            {"boxes": None, "scores": []}, [], 100
        )
        self.assertEqual(out, [])

    def test_filters_scores_outside_zero_one(self):
        result = {
            "boxes": [[0, 0, 10, 10], [5, 5, 15, 15]],
            "scores": [0.8, 1.2],
            "text_labels": ["cat", "dog"],
        }
        out = _normalize_results_owlvit(result, [["cat", "dog"]], 100)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["score"], 0.8)

    def test_filters_invalid_boxes(self):
        result = {
            "boxes": [[0, 0, 10, 10], [20, 20, 10, 10]],
            "scores": [0.9, 0.8],
            "text_labels": ["a", "b"],
        }
        out = _normalize_results_owlvit(result, [["a", "b"]], 100)
        self.assertEqual(len(out), 1)

    def test_output_schema(self):
        result = {
            "boxes": [[1, 2, 11, 12]],
            "scores": [0.95],
            "text_labels": ["person"],
        }
        out = _normalize_results_owlvit(result, [["person"]], 100)
        self.assertEqual(len(out), 1)
        d = out[0]
        self.assertEqual(d["label"], "person")
        self.assertEqual(d["score"], 0.95)
        self.assertEqual(d["box"]["format"], "xyxy")
        self.assertEqual(d["box"]["xmin"], 1)


class TestOwlvitConfig(unittest.TestCase):
    """Config validation for open-vocabulary (OWL-ViT) task."""

    def test_normalize_config_open_vocabulary_requires_text_labels(self):
        with self.assertRaises(ValueError) as ctx:
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="google/owlvit-base-patch32",
                    revision="main",
                    detection_type="open-vocabulary",
                )
            )
        self.assertIn("text_labels", str(ctx.exception).lower())

    def test_normalize_config_accepts_open_vocabulary_with_text_labels(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="google/owlvit-base-patch32",
                revision="main",
                detection_type="open-vocabulary",
                text_labels=[["a photo of a cat", "a photo of a dog"]],
            )
        )
        self.assertEqual(config.detection_type, "open-vocabulary")
        self.assertEqual(len(config.text_labels), 1)
        self.assertIn("cat", config.text_labels[0][0])


if __name__ == "__main__":
    unittest.main()
