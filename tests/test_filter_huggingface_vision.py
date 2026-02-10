#!/usr/bin/env python
"""Tests for CI: config validation and normalization logic (no model download)."""

import unittest

from filter_huggingface_vision.backends.grounding_dino import (
    _normalize_results as _normalize_results_grounding,
)
from filter_huggingface_vision.backends.object_detection import _normalize_detections
from filter_huggingface_vision.backends.owlvit import (
    _normalize_results as _normalize_results_owlvit,
)
from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)


class TestNormalizeDetections(unittest.TestCase):
    """Unit tests for _normalize_detections (object_detection backend). No model download."""

    def test_empty_results_returns_empty_list(self):
        model_config = type("Config", (), {"id2label": {}})()
        out = _normalize_detections(
            {"scores": None, "labels": [], "boxes": []}, model_config, 100
        )
        self.assertEqual(out, [])

    def test_missing_scores_returns_empty_list(self):
        model_config = type("Config", (), {"id2label": {}})()
        out = _normalize_detections(
            {"labels": [0], "boxes": [[0, 0, 10, 10]]}, model_config, 100
        )
        self.assertEqual(out, [])

    def test_filters_scores_outside_zero_one(self):
        model_config = type("Config", (), {"id2label": {0: "cat"}})()
        results = {
            "scores": [0.9, 1.5, -0.1],
            "labels": [0, 0, 0],
            "boxes": [[0, 0, 10, 10], [5, 5, 15, 15], [10, 10, 20, 20]],
        }
        out = _normalize_detections(results, model_config, 100)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["score"], 0.9)
        self.assertEqual(out[0]["label"], "cat")

    def test_filters_invalid_boxes(self):
        model_config = type("Config", (), {"id2label": {0: "x"}})()
        results = {
            "scores": [0.9, 0.8],
            "labels": [0, 0],
            "boxes": [[0, 0, 10, 10], [15, 15, 10, 10]],
        }
        out = _normalize_detections(results, model_config, 100)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["box"]["xmin"], 0)
        self.assertEqual(out[0]["box"]["xmax"], 10)

    def test_respects_max_detections(self):
        model_config = type("Config", (), {"id2label": {i: str(i) for i in range(10)}})()
        results = {
            "scores": [0.9 - i * 0.1 for i in range(10)],
            "labels": list(range(10)),
            "boxes": [[i, i, i + 10, i + 10] for i in range(10)],
        }
        out = _normalize_detections(results, model_config, max_detections=3)
        self.assertEqual(len(out), 3)

    def test_output_schema(self):
        model_config = type("Config", (), {"id2label": {0: "dog"}})()
        results = {
            "scores": [0.95],
            "labels": [0],
            "boxes": [[1.0, 2.0, 11.0, 12.0]],
        }
        out = _normalize_detections(results, model_config, 100)
        self.assertEqual(len(out), 1)
        d = out[0]
        self.assertIn("label", d)
        self.assertIn("score", d)
        self.assertIn("box", d)
        self.assertEqual(d["label"], "dog")
        self.assertEqual(d["score"], 0.95)
        self.assertEqual(d["box"]["format"], "xyxy")
        self.assertEqual(d["box"]["xmin"], 1.0)
        self.assertEqual(d["box"]["ymin"], 2.0)
        self.assertEqual(d["box"]["xmax"], 11.0)
        self.assertEqual(d["box"]["ymax"], 12.0)


class TestNormalizeResultsOwlvit(unittest.TestCase):
    """Unit tests for _normalize_results (owlvit backend). No model download."""

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


class TestNormalizeResultsGrounding(unittest.TestCase):
    """Unit tests for _normalize_results (grounding_dino backend). No model download."""

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


class TestFilterHuggingfaceVision(unittest.TestCase):
    """Fast tests: config validation only. No model download."""

    def test_normalize_config_rejects_empty_revision(self):
        with self.assertRaises(ValueError) as ctx:
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="PekingU/rtdetr_r50vd",
                    revision="",
                )
            )
        self.assertIn("revision", str(ctx.exception).lower())

    def test_normalize_config_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="x",
                    revision="main",
                    threshold=1.5,
                )
            )

    def test_normalize_config_rejects_unknown_detection_type(self):
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="x",
                    revision="main",
                    detection_type="image-classification",
                )
            )

    def test_normalize_config_accepts_valid_config(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="PekingU/rtdetr_r50vd",
                revision="main",
                threshold=0.3,
            )
        )
        self.assertEqual(config.model_id, "PekingU/rtdetr_r50vd")
        self.assertEqual(config.revision, "main")
        self.assertEqual(config.detection_type, "closed-vocabulary")
        self.assertEqual(config.threshold, 0.3)

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
