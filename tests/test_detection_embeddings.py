"""Crop embedding gates and attachment. No model download."""

import unittest

from PIL import Image

from filter_huggingface_vision.backends.embedding import _forward_embedding
from filter_huggingface_vision.detection_embeddings import (
    attach_crop_embeddings,
    l2_normalize,
    square_box,
)
from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
    _payload_to_meta_format,
)


def _filled(size):
    image = Image.new("RGB", (size, size))
    pixels = []
    for y in range(size):
        for x in range(size):
            value = 255 if (x // 4 + y // 4) % 2 == 0 else 0
            pixels.append((value, value, value))
    image.putdata(pixels)
    return image


class _Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _config(**overrides):
    base = {
        "embed_classes": ["face"],
        "embedding_model_id": "identity-model",
        "appearance_model_id": "",
        "embed_min_score": 0.5,
        "embed_min_bbox_area": 100,
        "embed_reject_border": True,
        "embed_min_sharpness": 1.0,
        "embed_bbox_expansion": 0.0,
        "embed_align_crops": False,
    }
    base.update(overrides)
    return _Config(base)


def _detection(label="face", score=0.9, box=None):
    return {
        "label": label,
        "score": score,
        "box": box or {"xmin": 16, "ymin": 16, "xmax": 48, "ymax": 48},
    }


class TestIdentityForward(unittest.TestCase):
    def test_named_embedding_wins_over_hidden_state(self):
        import torch

        class Output:
            embeddings = torch.tensor([[3.0, 4.0]])
            last_hidden_state = torch.tensor([[[1.0, 0.0]]])

        raw = _forward_embedding(Output())
        self.assertEqual(raw.shape, (1, 2))
        self.assertEqual(float(raw[0, 0]), 3.0)


class TestL2(unittest.TestCase):
    def test_unit_length(self):
        vector = l2_normalize([3.0, 4.0])
        self.assertAlmostEqual(sum(v * v for v in vector), 1.0, places=5)


class TestAttach(unittest.TestCase):
    def test_two_vectors_from_one_crop_decision(self):
        seen = []

        def primary(image):
            seen.append(("primary", image.size))
            return [3.0, 4.0]

        def appearance(image):
            seen.append(("appearance", image.size))
            return [0.0, 5.0]

        image = _filled(64)
        detection = _detection()
        attach_crop_embeddings(image, [detection], _config(appearance_model_id="dino"), primary, appearance)
        self.assertEqual([item[0] for item in seen], ["primary", "appearance"])
        models = [item["model"] for item in detection["embeddings"]]
        self.assertEqual(models, ["identity-model", "dino"])
        for item in detection["embeddings"]:
            self.assertAlmostEqual(sum(v * v for v in item["vector"]), 1.0, places=5)

    def test_other_class_is_left_without_a_vector(self):
        detection = _detection(label="bag")
        attach_crop_embeddings(image=_filled(64), detections=[detection], config=_config(), primary=lambda image: [1.0], appearance=None)
        self.assertNotIn("embeddings", detection)

    def test_low_score_publishes_nothing(self):
        detection = _detection(score=0.1)
        attach_crop_embeddings(
            _filled(64),
            [detection],
            _config(appearance_model_id="dino"),
            lambda image: [1.0],
            lambda image: [1.0],
        )
        self.assertNotIn("embeddings", detection)

    def test_border_box_publishes_nothing(self):
        detection = _detection(box={"xmin": 0, "ymin": 16, "xmax": 40, "ymax": 48})
        attach_crop_embeddings(_filled(64), [detection], _config(), lambda image: [1.0], None)
        self.assertNotIn("embeddings", detection)

    def test_align_squares_the_primary_crop(self):
        seen = []

        def primary(image):
            seen.append(image.size)
            return [1.0, 0.0]

        image = _filled(80)
        detection = _detection(box={"xmin": 20, "ymin": 30, "xmax": 50, "ymax": 40})
        attach_crop_embeddings(
            image,
            [detection],
            _config(embed_align_crops=True, embed_min_bbox_area=10),
            primary,
            None,
        )
        self.assertEqual(seen[0][0], seen[0][1])

    def test_square_box_is_centered(self):
        squared = square_box((10, 20, 30, 30), 100, 100)
        self.assertAlmostEqual(squared[2] - squared[0], squared[3] - squared[1])


class TestMeta(unittest.TestCase):
    def test_embeddings_are_copied_onto_the_published_detection(self):
        payload = {
            "detections": [
                {
                    "label": "face",
                    "score": 0.9,
                    "box": {"xmin": 1, "ymin": 1, "xmax": 4, "ymax": 4},
                    "embeddings": [{"model": "identity-model", "vector": [1.0, 0.0]}],
                }
            ]
        }
        detections, _, _ = _payload_to_meta_format(payload, 10, 10, "normalized")
        self.assertEqual(detections[0]["embeddings"][0]["model"], "identity-model")
        self.assertEqual(detections[0]["class"], "face")


class TestConfig(unittest.TestCase):
    def test_detections_scope_requires_classes_and_model(self):
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="detector",
                    revision="main",
                    embed_scope="detections",
                )
            )

    def test_embedding_type_rejects_detection_scope(self):
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="facebook/dinov2-small",
                    revision="main",
                    detection_type="embedding",
                    embed_scope="both",
                    embedding_model_id="facebook/dinov2-small",
                    embed_classes=["face"],
                )
            )
