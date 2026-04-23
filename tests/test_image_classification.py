#!/usr/bin/env python
"""Tests for image classification task (no model download)."""

import unittest
from unittest.mock import MagicMock, patch

from filter_huggingface_vision.backends.image_classification import (
    _logits_to_classifications,
)
from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)


class TestLogitsToClassifications(unittest.TestCase):
    """Unit tests for _logits_to_classifications (image_classification backend)."""

    def test_empty_logits_returns_empty_list(self):
        import torch

        out = _logits_to_classifications(None, {0: "a"}, 5)
        self.assertEqual(out, [])
        logits = torch.zeros(0, 10)
        out = _logits_to_classifications(logits, {i: str(i) for i in range(10)}, 5)
        self.assertEqual(out, [])

    def test_respects_top_k(self):
        import torch

        logits = torch.tensor([[1.0, 2.0, 0.0, 0.5, 3.0]])
        id2label = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
        out = _logits_to_classifications(logits, id2label, top_k=2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["label"], "e")
        self.assertEqual(out[1]["label"], "b")

    def test_sorted_by_score_descending(self):
        import torch

        logits = torch.tensor([[0.0, 1.0, 2.0]])
        id2label = {0: "a", 1: "b", 2: "c"}
        out = _logits_to_classifications(logits, id2label, top_k=3)
        self.assertEqual(len(out), 3)
        self.assertGreaterEqual(out[0]["score"], out[1]["score"])
        self.assertGreaterEqual(out[1]["score"], out[2]["score"])
        self.assertEqual(out[0]["label"], "c")
        self.assertEqual(out[1]["label"], "b")
        self.assertEqual(out[2]["label"], "a")

    def test_output_schema(self):
        import torch

        logits = torch.tensor([[0.0, 1.0]])
        id2label = {0: "x", 1: "y"}
        out = _logits_to_classifications(logits, id2label, top_k=2)
        self.assertEqual(len(out), 2)
        for d in out:
            self.assertIn("label", d)
            self.assertIn("score", d)
            self.assertIsInstance(d["label"], str)
            self.assertIsInstance(d["score"], float)
            self.assertGreaterEqual(d["score"], 0.0)
            self.assertLessEqual(d["score"], 1.0)

    def test_unknown_label_id_uses_str_of_id(self):
        import torch

        logits = torch.tensor([[1.0, 0.0]])
        id2label = {0: "known"}
        out = _logits_to_classifications(logits, id2label, top_k=2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["label"], "known")
        self.assertEqual(out[1]["label"], "1")


class TestImageClassificationFilter(unittest.TestCase):
    """Filter process() with image-classification backend (mock, no model download)."""

    def test_process_puts_classifications_in_payload_and_empty_detections(self):
        from PIL import Image

        class MockClassificationBackend:
            def load(self, config):
                pass

            def run(self, image_pil, width, height, config):
                return {
                    "classifications": [
                        {"label": "tabby_cat", "score": 0.92},
                        {"label": "Egyptian_cat", "score": 0.05},
                    ]
                }

            def shutdown(self):
                pass

        def get_mock_backend(detection_type):
            assert detection_type == "image-classification"
            return MockClassificationBackend

        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="google/vit-base-patch16-224",
                revision="main",
                detection_type="image-classification",
                top_k=5,
            )
        )
        pil_image = Image.new("RGB", (224, 224), color="red")
        frame = type("Frame", (), {"data": {"main": pil_image}})()

        with patch(
            "filter_huggingface_vision.filter.get_backend",
            side_effect=get_mock_backend,
        ):
            filter_inst = FilterHuggingfaceVision(config)
            filter_inst.setup(config)

        try:
            frames = {"main": frame}
            out = filter_inst.process(frames)
        finally:
            filter_inst.shutdown()

        self.assertIn("main", out)
        self.assertIn("meta", frame.data)
        meta = frame.data["meta"]
        self.assertNotIn("detections", meta)
        self.assertNotIn("detection_confidence", meta)
        self.assertIn("classification", meta)
        cl = meta["classification"]
        self.assertEqual(cl["architecture"], "huggingface")
        self.assertEqual(cl["classes"], ["tabby_cat", "Egyptian_cat"])
        self.assertEqual(cl["confidences"], [0.92, 0.05])
        self.assertIn("timestamp", cl)
        self.assertEqual(cl["filter_id"], "test")
        self.assertEqual(cl["model_id"], "google/vit-base-patch16-224")
        self.assertEqual(cl["revision"], "main")
        self.assertEqual(cl["top_k"], 5)


class TestImageClassificationConfig(unittest.TestCase):
    """Config validation for image-classification task."""

    def test_normalize_config_accepts_image_classification(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="google/vit-base-patch16-224",
                revision="main",
                detection_type="image-classification",
                top_k=5,
            )
        )
        self.assertEqual(config.detection_type, "image-classification")
        self.assertEqual(config.model_id, "google/vit-base-patch16-224")
        self.assertEqual(config.revision, "main")
        self.assertEqual(config.top_k, 5)

    def test_normalize_config_image_classification_does_not_require_text_labels(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="facebook/convnext-tiny-224",
                revision="main",
                detection_type="image-classification",
            )
        )
        self.assertEqual(config.detection_type, "image-classification")
        self.assertIsNone(config.text_labels)

    def test_normalize_config_image_classification_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError) as ctx:
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="google/vit-base-patch16-224",
                    revision="main",
                    detection_type="image-classification",
                    top_k=0,
                )
            )
        self.assertIn("top_k", str(ctx.exception).lower())
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="google/vit-base-patch16-224",
                    revision="main",
                    detection_type="image-classification",
                    top_k=1001,
                )
            )


class TestImageClassificationBackendLoadErrors(unittest.TestCase):
    """Unit tests for structured error messages on ImageClassificationBackend.load() failures."""

    _CONFIG = {"model_id": "org/model", "revision": "abc123", "device": "cpu"}

    def _load_backend(self):
        from filter_huggingface_vision.backends.image_classification import (
            ImageClassificationBackend,
        )

        ImageClassificationBackend().load(self._CONFIG)

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

    def _make_hf_error(self, cls, message):
        mock_response = MagicMock()
        mock_response.status_code = 404
        return cls(message, response=mock_response)

    def test_repository_not_found_error_message(self):
        from huggingface_hub import errors as _hf_errors

        exc = self._make_hf_error(_hf_errors.RepositoryNotFoundError, "org/model")
        with patch("transformers.AutoImageProcessor.from_pretrained", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("org/model", msg)
        self.assertIn("not found", msg)

    def test_revision_not_found_error_message(self):
        from huggingface_hub import errors as _hf_errors

        exc = self._make_hf_error(_hf_errors.RevisionNotFoundError, "abc123")
        with patch("transformers.AutoImageProcessor.from_pretrained", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("abc123", msg)
        self.assertIn("Revision", msg)

    def test_gated_repo_error_message(self):
        from huggingface_hub import errors as _hf_errors

        exc = self._make_hf_error(_hf_errors.GatedRepoError, "org/model")
        with patch("transformers.AutoImageProcessor.from_pretrained", side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("license", msg)

    def test_hf_hub_http_error_includes_repr(self):
        from huggingface_hub import errors as _hf_errors

        exc = self._make_hf_error(_hf_errors.HfHubHTTPError, "503 Service Unavailable")
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
        self.assertIn("image-classification", msg)
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
