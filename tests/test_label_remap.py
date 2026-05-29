#!/usr/bin/env python
"""Tests for configurable class-name remapping (PLAT-1104). No model download."""

import unittest
from unittest import mock

import numpy as np

from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
    _apply_label_map,
    _create_visualization,
    _parse_text_labels,
    _payload_to_meta_format,
)


class TestParseTextLabels(unittest.TestCase):
    """_parse_text_labels(value, class_delimiter, prompt_delimiter) -> (prompts, label_map)."""

    def test_none_returns_none_and_empty_map(self):
        prompts, label_map = _parse_text_labels(None, "|||", "###")
        self.assertIsNone(prompts)
        self.assertEqual(label_map, {})

    def test_bare_prompts_map_to_self(self):
        prompts, label_map = _parse_text_labels("car###truck###dog", "|||", "###")
        self.assertEqual(prompts, [["car", "truck", "dog"]])
        self.assertEqual(label_map, {"car": "car", "truck": "truck", "dog": "dog"})

    def test_inline_mapping(self):
        prompts, label_map = _parse_text_labels(
            "vehicle|||car###vehicle|||truck###animal|||dog", "|||", "###"
        )
        self.assertEqual(prompts, [["car", "truck", "dog"]])
        self.assertEqual(
            label_map, {"car": "vehicle", "truck": "vehicle", "dog": "animal"}
        )

    def test_mixed_mapped_and_bare(self):
        prompts, label_map = _parse_text_labels(
            "vehicle|||car###truck###animal|||dog", "|||", "###"
        )
        self.assertEqual(prompts, [["car", "truck", "dog"]])
        self.assertEqual(
            label_map, {"car": "vehicle", "truck": "truck", "dog": "animal"}
        )

    def test_whitespace_trimmed(self):
        prompts, label_map = _parse_text_labels(
            "  vehicle|||car  ###  animal|||dog ### truck  ", "|||", "###"
        )
        self.assertEqual(prompts, [["car", "dog", "truck"]])
        self.assertEqual(
            label_map, {"car": "vehicle", "dog": "animal", "truck": "truck"}
        )

    def test_custom_delimiters(self):
        prompts, label_map = _parse_text_labels(
            "vehicle=car|animal=dog|truck", "=", "|"
        )
        self.assertEqual(prompts, [["car", "dog", "truck"]])
        self.assertEqual(
            label_map, {"car": "vehicle", "dog": "animal", "truck": "truck"}
        )

    def test_existing_list_passthrough(self):
        prompts, label_map = _parse_text_labels(
            [["a person", "a cup"]], "|||", "###"
        )
        self.assertEqual(prompts, [["a person", "a cup"]])
        self.assertEqual(label_map, {})

    def test_duplicate_prompt_mapping_raises(self):
        with self.assertRaises(ValueError):
            _parse_text_labels("vehicle|||car###automobile|||car", "|||", "###")

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            _parse_text_labels({"car": "vehicle"}, "|||", "###")

    def test_gun_scenario(self):
        prompts, label_map = _parse_text_labels(
            "gun|||a handgun###gun|||a shotgun", "|||", "###"
        )
        self.assertEqual(prompts, [["a handgun", "a shotgun"]])
        self.assertEqual(label_map, {"a handgun": "gun", "a shotgun": "gun"})


class TestApplyLabelMap(unittest.TestCase):
    """_apply_label_map(detections, label_map, collapse_labels_to)."""

    def _dets(self):
        return [
            {"label": "a handgun", "score": 0.5, "box": {}},
            {"label": "person", "score": 0.4, "box": {}},
        ]

    def test_label_map_renames(self):
        out = _apply_label_map(self._dets(), {"person": "people"}, None)
        self.assertEqual([d["label"] for d in out], ["a handgun", "people"])

    def test_collapse_overrides_everything(self):
        out = _apply_label_map(self._dets(), {"person": "people"}, "weapon")
        self.assertEqual([d["label"] for d in out], ["weapon", "weapon"])

    def test_unmapped_passthrough(self):
        out = _apply_label_map(self._dets(), {}, None)
        self.assertEqual([d["label"] for d in out], ["a handgun", "person"])

    def test_none_map_passthrough(self):
        out = _apply_label_map(self._dets(), None, None)
        self.assertEqual([d["label"] for d in out], ["a handgun", "person"])


class TestNormalizeConfigRemap(unittest.TestCase):
    """normalize_config wiring + validation for remap fields."""

    def _cfg(self, **overrides):
        base = dict(
            id="test",
            sources="",
            outputs="",
            model_id="google/owlvit-base-patch32",
            revision="main",
            detection_type="open-vocabulary",
        )
        base.update(overrides)
        return FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(**base)
        )

    def test_inline_string_normalizes_text_labels_and_builds_map(self):
        cfg = self._cfg(text_labels="gun|||a handgun###gun|||a shotgun")
        self.assertEqual(getattr(cfg, "text_labels"), [["a handgun", "a shotgun"]])
        self.assertEqual(
            getattr(cfg, "label_map"), {"a handgun": "gun", "a shotgun": "gun"}
        )

    def test_explicit_label_map_merges_and_wins(self):
        cfg = self._cfg(
            text_labels="gun|||a handgun",
            label_map={"a handgun": "firearm", "person": "people"},
        )
        # explicit wins on key conflict
        self.assertEqual(getattr(cfg, "label_map")["a handgun"], "firearm")
        self.assertEqual(getattr(cfg, "label_map")["person"], "people")

    def test_empty_delimiter_raises(self):
        with self.assertRaises(ValueError):
            self._cfg(text_labels="gun|||a handgun", class_delimiter="")

    def test_equal_delimiters_raise(self):
        with self.assertRaises(ValueError):
            self._cfg(
                text_labels="gun|||a handgun",
                class_delimiter="###",
                prompt_delimiter="###",
            )

    def test_collapse_must_be_string(self):
        with self.assertRaises(ValueError):
            self._cfg(text_labels=[["a handgun"]], collapse_labels_to=123)


class TestMetaAndVizAgree(unittest.TestCase):
    """After remap, meta class and visualization overlay use the same final name."""

    def _payload(self):
        return {
            "detection_type": "open-vocabulary",
            "task": "zero-shot-object-detection",
            "model": {"id": "x", "revision": "main"},
            "image": {"width": 100, "height": 100},
            "detections": [
                {
                    "label": "a handgun",
                    "score": 0.5,
                    "box": {"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50},
                }
            ],
        }

    def test_meta_class_uses_remapped_label(self):
        payload = self._payload()
        _apply_label_map(payload["detections"], {"a handgun": "gun"}, None)
        dets, conf, _ = _payload_to_meta_format(payload, 100, 100)
        self.assertEqual(dets[0]["class"], "gun")

    def test_visualization_draws_remapped_label(self):
        payload = self._payload()
        _apply_label_map(payload["detections"], None, "weapon")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with mock.patch("cv2.putText") as put_text:
            _create_visualization(image, payload)
        drawn = " ".join(str(c.args[1]) for c in put_text.call_args_list)
        self.assertIn("weapon", drawn)
        self.assertNotIn("handgun", drawn)


if __name__ == "__main__":
    unittest.main()
