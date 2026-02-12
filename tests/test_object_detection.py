#!/usr/bin/env python
"""Tests for closed-vocabulary object detection task (no model download)."""

import unittest

from filter_huggingface_vision.backends.object_detection import _normalize_detections
from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)


class TestNormalizeDetections(unittest.TestCase):
    """Unit tests for _normalize_detections (object_detection backend)."""

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


class TestObjectDetectionConfig(unittest.TestCase):
    """Config validation for closed-vocabulary object detection."""

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


if __name__ == "__main__":
    unittest.main()
