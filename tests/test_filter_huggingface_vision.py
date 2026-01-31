#!/usr/bin/env python
"""Simple tests for CI: config validation only (no model download)."""

import os
import unittest

from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)


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


if __name__ == "__main__":
    unittest.main()
