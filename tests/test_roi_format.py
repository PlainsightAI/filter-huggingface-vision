#!/usr/bin/env python
"""Tests for the roi_format config (normalized [0,1] vs integer pixel coords)."""

import unittest

from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
    _payload_to_meta_format,
)


def _payload():
    return {
        "detections": [
            {
                "label": "gun",
                "score": 0.9,
                "box": {"xmin": 100.4, "ymin": 50.6, "xmax": 200.5, "ymax": 150.0},
            }
        ]
    }


class TestPayloadToMetaFormatRoiFormat(unittest.TestCase):
    def test_default_is_normalized(self):
        rois = _payload_to_meta_format(_payload(), 400, 300)[0][0]["rois"]
        self.assertEqual(rois, [[100.4 / 400, 50.6 / 300, 200.5 / 400, 150.0 / 300]])

    def test_explicit_normalized_matches_default(self):
        default = _payload_to_meta_format(_payload(), 400, 300)[0][0]["rois"]
        explicit = _payload_to_meta_format(_payload(), 400, 300, "normalized")[0][0]["rois"]
        self.assertEqual(default, explicit)

    def test_pixel_emits_rounded_ints(self):
        rois = _payload_to_meta_format(_payload(), 400, 300, "pixel")[0][0]["rois"]
        self.assertEqual(rois, [[100, 51, 200, 150]])
        for v in rois[0]:
            self.assertIsInstance(v, int)

    def test_pixel_ignores_width_height(self):
        # Pixel coords must not depend on the frame size passed in.
        a = _payload_to_meta_format(_payload(), 400, 300, "pixel")[0][0]["rois"]
        b = _payload_to_meta_format(_payload(), 1920, 1080, "pixel")[0][0]["rois"]
        self.assertEqual(a, b)

    def test_confidence_unaffected_by_format(self):
        norm = _payload_to_meta_format(_payload(), 400, 300)[1]
        px = _payload_to_meta_format(_payload(), 400, 300, "pixel")[1]
        self.assertEqual(norm, px)
        self.assertAlmostEqual(px, 0.9)


class TestRoiFormatConfigValidation(unittest.TestCase):
    def _normalize(self, roi_format):
        return FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="t",
                sources="",
                outputs="",
                model_id="m",
                revision="main",
                detection_type="closed-vocabulary",
                roi_format=roi_format,
            )
        )

    def test_default_when_unset_is_normalized(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="t",
                sources="",
                outputs="",
                model_id="m",
                revision="main",
                detection_type="closed-vocabulary",
            )
        )
        self.assertEqual(getattr(config, "roi_format", "normalized"), "normalized")

    def test_accepts_normalized_and_pixel(self):
        self.assertEqual(self._normalize("normalized").roi_format, "normalized")
        self.assertEqual(self._normalize("pixel").roi_format, "pixel")

    def test_rejects_unknown_value(self):
        for bad in ("PIXEL", "pixels", "foo"):
            with self.assertRaises(ValueError) as ctx:
                self._normalize(bad)
            self.assertIn("roi_format", str(ctx.exception))

    def test_empty_falls_back_to_normalized(self):
        # Empty string defaults to "normalized", matching the `or` fallback style
        # used for the other config fields (detection_type, device, ...).
        self.assertEqual(self._normalize("").roi_format, "normalized")


if __name__ == "__main__":
    unittest.main()
