#!/usr/bin/env python
"""Tests for closed-vocabulary object detection task (no model download)."""

import unittest
from unittest.mock import MagicMock, patch

from filter_huggingface_vision.backends.object_detection import _normalize_detections
from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)
from tests._hf_test_utils import make_hf_error


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


class TestObjectDetectionBackendLoadErrors(unittest.TestCase):
    """Unit tests for structured error messages on ObjectDetectionBackend.load() failures."""

    _CONFIG = {"model_id": "org/model", "revision": "abc123", "device": "cpu"}

    def _load_backend(self):
        from filter_huggingface_vision.backends.object_detection import (
            ObjectDetectionBackend,
        )

        ObjectDetectionBackend().load(self._CONFIG)

    # --- ImportError branches ---

    def test_timm_import_error_gives_actionable_message(self):
        with patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=ImportError("No module named 'timm'"),
        ):
            with self.assertRaises(ImportError) as ctx:
                self._load_backend()
        self.assertIn("timm", str(ctx.exception))
        self.assertIn("pip install timm", str(ctx.exception))

    def test_non_timm_import_error_is_reraised_unchanged(self):
        original = ImportError("No module named 'some_other_dep'")
        with patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=original,
        ):
            with self.assertRaises(ImportError) as ctx:
                self._load_backend()
        self.assertIs(ctx.exception, original)

    # --- HuggingFace Hub error branches ---

    def test_repository_not_found_error_message(self):
        from huggingface_hub import errors as _hf_errors

        exc = make_hf_error(_hf_errors.RepositoryNotFoundError, "org/model")
        with patch("transformers.AutoImageProcessor.from_pretrained", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("org/model", msg)
        self.assertIn("not found", msg)

    def test_repository_not_found_on_model_download_gives_actionable_message(self):
        from huggingface_hub import errors as _hf_errors

        exc = make_hf_error(_hf_errors.RepositoryNotFoundError, "org/model")
        with patch(
            "transformers.AutoImageProcessor.from_pretrained",
            return_value=MagicMock(),
        ):
            with patch(
                "transformers.AutoModelForObjectDetection.from_pretrained",
                side_effect=exc,
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("org/model", msg)
        self.assertIn("not found", msg)
        self.assertIs(ctx.exception.__cause__, exc)

    def test_revision_not_found_error_message(self):
        from huggingface_hub import errors as _hf_errors

        exc = make_hf_error(_hf_errors.RevisionNotFoundError, "abc123")
        with patch("transformers.AutoImageProcessor.from_pretrained", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("abc123", msg)
        self.assertIn("Revision", msg)

    def test_gated_repo_error_message(self):
        from huggingface_hub import errors as _hf_errors

        exc = make_hf_error(_hf_errors.GatedRepoError, "org/model")
        with patch("transformers.AutoImageProcessor.from_pretrained", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("license", msg)

    def test_hf_hub_http_error_includes_repr(self):
        from huggingface_hub import errors as _hf_errors

        exc = make_hf_error(_hf_errors.HfHubHTTPError, "503 Service Unavailable")
        with patch("transformers.AutoImageProcessor.from_pretrained", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("org/model", msg)
        self.assertIn("HuggingFace Hub", msg)

    # --- ValueError / config-parse branch ---

    def test_value_error_gives_incompatibility_message_with_repr(self):
        with patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=ValueError("unrecognized architecture"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("object detection", msg)
        self.assertIn("unrecognized architecture", msg)

    # --- Fallback branch ---

    def test_unexpected_exception_includes_repr(self):
        with patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("Unexpected", msg)
        self.assertIn("disk full", msg)

    # --- Chained cause is preserved ---

    def test_original_cause_is_chained(self):
        original = ValueError("root cause")
        with patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=original,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        self.assertIs(ctx.exception.__cause__, original)


if __name__ == "__main__":
    unittest.main()
