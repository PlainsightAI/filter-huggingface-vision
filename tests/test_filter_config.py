#!/usr/bin/env python
"""Shared filter config validation tests (no model download)."""

import unittest

from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)


class TestFilterConfig(unittest.TestCase):
    """Config validation shared across all detection types."""

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
                    detection_type="unknown-task",
                )
            )


if __name__ == "__main__":
    unittest.main()
