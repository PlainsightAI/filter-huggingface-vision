"""Crop embedding gates and attachment. No model download."""

import unittest

from PIL import Image

from filter_huggingface_vision.backends.embedding import _recognition_vector
from filter_huggingface_vision.detection_embeddings import (
    DEFAULT_EMBED_MIN_BBOX_AREA,
    DEFAULT_EMBED_MIN_SHARPNESS,
    attach_crop_embeddings,
    l2_normalize,
    resolve_embedder_revision,
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
            pooler_output = torch.tensor([[9.0, 9.0]])
            logits = torch.tensor([[0.1, 0.2, 0.3]])

        raw = _recognition_vector(Output())
        self.assertEqual(tuple(raw.shape), (1, 2))
        self.assertEqual(float(raw[0, 0]), 3.0)

    def test_raw_vector_is_accepted(self):
        import torch

        raw = _recognition_vector(torch.tensor([3.0, 4.0]))
        self.assertEqual(tuple(raw.shape), (1, 2))

    def test_logits_and_feature_maps_are_refused(self):
        import torch

        class Logits:
            logits = torch.tensor([[0.1, 0.2, 0.3]])

        class Features:
            last_hidden_state = torch.tensor([[[1.0, 0.0]]])
            pooler_output = torch.tensor([[1.0, 0.0]])

        with self.assertRaises(ValueError):
            _recognition_vector(Logits())
        with self.assertRaises(ValueError):
            _recognition_vector(Features())
        with self.assertRaises(ValueError):
            _recognition_vector(torch.zeros(1, 4, 4))


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

    def test_align_embeds_the_warped_face_and_keeps_the_appearance_crop(self):
        seen_primary = []
        seen_appearance = []

        def primary(image):
            seen_primary.append(image.size)
            return [1.0, 0.0]

        def appearance(image):
            seen_appearance.append(image.size)
            return [0.0, 1.0]

        class _Aligner:
            def align(self, image, boxes):
                return [Image.new("RGB", (112, 112)) for _ in boxes]

        image = _filled(80)
        detection = _detection(box={"xmin": 20, "ymin": 30, "xmax": 50, "ymax": 40})
        attach_crop_embeddings(
            image,
            [detection],
            _config(embed_align_crops=True, embed_min_bbox_area=10, appearance_model_id="dino"),
            primary,
            appearance,
            _Aligner(),
        )
        self.assertEqual(seen_primary, [(112, 112)])
        self.assertEqual(seen_appearance, [(30, 10)])
        self.assertEqual(len(detection["embeddings"]), 2)

    def test_align_without_landmarks_publishes_nothing(self):
        called = []

        class _Aligner:
            def align(self, image, boxes):
                return [None for _ in boxes]

        detection = _detection(box={"xmin": 20, "ymin": 30, "xmax": 50, "ymax": 40})
        attach_crop_embeddings(
            _filled(80),
            [detection],
            _config(embed_align_crops=True, embed_min_bbox_area=10, embed_min_sharpness=0),
            lambda image: called.append(image.size) or [1.0, 0.0],
            None,
            _Aligner(),
        )
        self.assertNotIn("embeddings", detection)
        self.assertEqual(called, [])

    def test_align_refuses_a_non_square_crop(self):
        class _Aligner:
            def align(self, image, boxes):
                return [Image.new("RGB", (30, 10)) for _ in boxes]

        detection = _detection(box={"xmin": 20, "ymin": 30, "xmax": 50, "ymax": 40})
        attach_crop_embeddings(
            _filled(80),
            [detection],
            _config(embed_align_crops=True, embed_min_bbox_area=10, embed_min_sharpness=0),
            lambda image: [1.0, 0.0],
            None,
            _Aligner(),
        )
        self.assertNotIn("embeddings", detection)

    def test_embed_max_detections_keeps_the_highest_scores(self):
        seen = []

        def primary(image):
            seen.append(image.size)
            return [1.0, 0.0]

        image = _filled(120)
        detections = [
            _detection(score=0.55, box={"xmin": 4, "ymin": 4, "xmax": 30, "ymax": 30}),
            _detection(score=0.95, box={"xmin": 40, "ymin": 4, "xmax": 70, "ymax": 30}),
            _detection(score=0.80, box={"xmin": 74, "ymin": 4, "xmax": 110, "ymax": 30}),
        ]
        attach_crop_embeddings(
            image,
            detections,
            _config(embed_min_bbox_area=10, embed_min_sharpness=0, embed_max_detections=2),
            primary,
            None,
        )
        self.assertEqual(len(seen), 2)
        self.assertIn("embeddings", detections[1])
        self.assertIn("embeddings", detections[2])
        self.assertNotIn("embeddings", detections[0])

    def test_align_requires_an_aligner(self):
        detection = _detection(box={"xmin": 20, "ymin": 20, "xmax": 50, "ymax": 50})
        with self.assertRaises(ValueError):
            attach_crop_embeddings(
                _filled(80),
                [detection],
                _config(embed_align_crops=True, embed_min_bbox_area=10),
                lambda images: [[1.0, 0.0]],
                None,
            )

    def test_passing_crops_share_one_forward_per_model(self):
        class _Batch:
            def __init__(self):
                self.counts = []

            def vectors(self, images):
                self.counts.append(len(images))
                return [[1.0, 0.0] for _ in images]

        primary = _Batch()
        appearance = _Batch()
        image = _filled(80)
        detections = [
            _detection(box={"xmin": 10, "ymin": 10, "xmax": 40, "ymax": 40}),
            _detection(box={"xmin": 42, "ymin": 12, "xmax": 70, "ymax": 40}),
        ]
        attach_crop_embeddings(
            image,
            detections,
            _config(appearance_model_id="dino", embed_min_bbox_area=10),
            primary,
            appearance,
        )
        self.assertEqual(primary.counts, [2])
        self.assertEqual(appearance.counts, [2])
        self.assertIn("embeddings", detections[0])
        self.assertIn("embeddings", detections[1])


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

    def test_quality_gates_default_on(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="detector",
                revision="main",
            )
        )
        self.assertEqual(config.embed_min_bbox_area, DEFAULT_EMBED_MIN_BBOX_AREA)
        self.assertEqual(config.embed_min_sharpness, DEFAULT_EMBED_MIN_SHARPNESS)
        self.assertGreater(config.embed_min_bbox_area, 0)
        self.assertGreater(config.embed_min_sharpness, 0)

    def test_other_models_do_not_inherit_the_detector_sha(self):
        detector_sha = "a" * 40
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="detector",
                revision=detector_sha,
                embed_scope="both",
                embedding_model_id="org/arcface",
                embedding_revision="face-sha",
                appearance_model_id="facebook/dinov2-small",
                appearance_revision="dino-sha",
                embed_classes=["face"],
            )
        )
        self.assertEqual(config.embedding_revision, "face-sha")
        self.assertEqual(config.appearance_revision, "dino-sha")
        self.assertEqual(
            resolve_embedder_revision("detector", None, "detector", detector_sha),
            detector_sha,
        )

    def test_a_different_embedder_must_name_its_own_revision(self):
        detector_sha = "a" * 40
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="IDEA-Research/grounding-dino-base",
                    revision=detector_sha,
                    embed_scope="detections",
                    embed_classes=["person"],
                    embedding_model_id="facebook/dinov2-base",
                )
            )

    def test_empty_embed_classes_are_rejected(self):
        for classes in ("", []):
            with self.assertRaises(ValueError):
                FilterHuggingfaceVision.normalize_config(
                    FilterHuggingfaceVisionConfig(
                        id="test",
                        sources="",
                        outputs="",
                        model_id="detector",
                        revision="main",
                        embed_scope="detections",
                        embed_classes=classes,
                        embedding_model_id="org/arcface",
                        embedding_revision="main",
                    )
                )

    def test_unused_appearance_model_is_rejected(self):
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="detector",
                    revision="main",
                    embed_scope="off",
                    appearance_model_id="facebook/dinov2-small",
                    appearance_revision="main",
                )
            )
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="facebook/dinov2-small",
                    revision="main",
                    detection_type="embedding",
                    appearance_model_id="facebook/dinov2-base",
                    appearance_revision="main",
                )
            )

    def test_bbox_area_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            FilterHuggingfaceVision.normalize_config(
                FilterHuggingfaceVisionConfig(
                    id="test",
                    sources="",
                    outputs="",
                    model_id="detector",
                    revision="main",
                    embed_min_bbox_area=0.5,
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
