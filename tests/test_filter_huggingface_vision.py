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

    def test_normalize_config_rejects_wrong_task(self):
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="x",
                    revision="main",
                    task="image-classification",
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
        self.assertEqual(config.task, "object-detection")
        self.assertEqual(config.threshold, 0.3)


if __name__ == "__main__":
    unittest.main()
