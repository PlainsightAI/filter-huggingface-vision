#!/usr/bin/env python
"""Tests for image classification task (no model download)."""

import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
