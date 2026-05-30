#!/usr/bin/env python
"""Tests for the two-stage ROI cascade (PLAT-1106) — pure helpers + config (no model download)."""

import unittest

from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
    _pad_and_clamp_region,
    _remap_box_to_full,
)


class TestRemapBoxToFull(unittest.TestCase):
    """Translate a crop-local pixel box to full-frame pixel coords."""

    def test_remap_translates_by_crop_origin(self):
        # crop origin at (100, 50); crop-local box (10, 20, 30, 40)
        out = _remap_box_to_full((10, 20, 30, 40), 100, 50)
        self.assertEqual(out, (110, 70, 130, 90))

    def test_remap_zero_origin_is_identity(self):
        out = _remap_box_to_full((1, 2, 3, 4), 0, 0)
        self.assertEqual(out, (1, 2, 3, 4))

    def test_remap_preserves_box_size(self):
        out = _remap_box_to_full((0, 0, 50, 25), 200, 300)
        self.assertEqual(out[2] - out[0], 50)
        self.assertEqual(out[3] - out[1], 25)


class TestPadAndClampRegion(unittest.TestCase):
    """Pad each side by a fraction of box w/h, clamp to frame, drop degenerate."""

    def test_pad_expands_each_side(self):
        # box 100x100 at (200,200); pad 0.3 -> +30 each side
        out = _pad_and_clamp_region((200, 200, 300, 300), 0.3, 1000, 1000)
        self.assertEqual(out, (170, 170, 330, 330))

    def test_pad_clamps_origin_at_zero(self):
        # box near top-left; padding would go negative -> clamp to 0
        out = _pad_and_clamp_region((10, 10, 110, 110), 0.3, 1000, 1000)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 0)

    def test_pad_clamps_max_at_frame(self):
        # box near bottom-right; padding would exceed W/H -> clamp to W/H
        out = _pad_and_clamp_region((900, 900, 990, 990), 0.3, 1000, 1000)
        self.assertEqual(out[2], 1000)
        self.assertEqual(out[3], 1000)

    def test_zero_pad_is_clamped_box(self):
        out = _pad_and_clamp_region((10.4, 20.6, 30.2, 40.9), 0.0, 1000, 1000)
        self.assertEqual(out, (10, 20, 30, 40))

    def test_degenerate_region_returns_none(self):
        # xmin >= xmax after clamping
        self.assertIsNone(_pad_and_clamp_region((50, 50, 50, 100), 0.0, 1000, 1000))
        self.assertIsNone(_pad_and_clamp_region((50, 50, 100, 50), 0.0, 1000, 1000))

    def test_box_fully_outside_frame_returns_none(self):
        # box entirely beyond the right edge clamps to a zero-width region
        self.assertIsNone(_pad_and_clamp_region((1100, 100, 1200, 200), 0.0, 1000, 1000))

    def test_returns_ints(self):
        out = _pad_and_clamp_region((200.0, 200.0, 300.0, 300.0), 0.1, 1000, 1000)
        self.assertTrue(all(isinstance(v, int) for v in out))


def _base_gate_config(**overrides):
    cfg = dict(
        id="test",
        sources="",
        outputs="",
        model_id="PekingU/rtdetr_r50vd",
        revision="main",
        gate_model_id="google/owlvit-base-patch32",
        gate_detection_type="open-vocabulary",
        gate_prompt=[["a photo of a person"]],
    )
    cfg.update(overrides)
    return FilterHuggingfaceVisionConfig(**cfg)


class TestGateConfigValidation(unittest.TestCase):
    """Validation + normalization of gate cascade config fields."""

    def test_open_vocab_gate_requires_gate_prompt(self):
        with self.assertRaises(ValueError) as ctx:
            FilterHuggingfaceVision.normalize_config(
                _base_gate_config(gate_prompt=None)
            )
        self.assertIn("gate_prompt", str(ctx.exception).lower())

    def test_closed_vocab_gate_does_not_require_prompt(self):
        config = FilterHuggingfaceVision.normalize_config(
            _base_gate_config(
                gate_detection_type="closed-vocabulary", gate_prompt=None
            )
        )
        self.assertEqual(config.gate_model_id, "google/owlvit-base-patch32")

    def test_bad_gate_pad_raises(self):
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(_base_gate_config(gate_pad=-0.1))

    def test_bad_gate_max_regions_raises(self):
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                _base_gate_config(gate_max_regions=0)
            )
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                _base_gate_config(gate_max_regions=2.5)
            )

    def test_gate_defaults_carried_through(self):
        config = FilterHuggingfaceVision.normalize_config(_base_gate_config())
        self.assertEqual(config.gate_detection_type, "open-vocabulary")
        self.assertEqual(config.gate_revision, "main")
        self.assertEqual(config.gate_pad, 0.3)
        self.assertEqual(config.gate_max_regions, 5)
        self.assertIsNone(config.gate_threshold)
        self.assertIsNone(config.gate_class)

    def test_gate_threshold_and_class_passthrough(self):
        config = FilterHuggingfaceVision.normalize_config(
            _base_gate_config(gate_threshold=0.15, gate_class="person")
        )
        self.assertEqual(config.gate_threshold, 0.15)
        self.assertEqual(config.gate_class, "person")


class TestSingleStagePassthrough(unittest.TestCase):
    """Regression guard: with no gate config, the single-stage path is unchanged."""

    def test_gate_model_id_defaults_to_none(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="PekingU/rtdetr_r50vd",
                revision="main",
            )
        )
        self.assertIsNone(config.gate_model_id)

    def test_gate_defaults_present_without_gate_active(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="PekingU/rtdetr_r50vd",
                revision="main",
            )
        )
        self.assertEqual(config.gate_detection_type, "closed-vocabulary")
        self.assertEqual(config.gate_pad, 0.3)
        self.assertEqual(config.gate_max_regions, 5)


if __name__ == "__main__":
    unittest.main()
