"""Local ArcFace IResNet. No Hub download and no remote code."""

import unittest

import torch
from PIL import Image

from filter_huggingface_vision.arcface import (
    ArcFaceEmbedder,
    build_iresnet,
    load_arcface_state,
)


class ArcFaceTests(unittest.TestCase):
    def test_ir101_forward_is_a_recognition_vector(self):
        model = build_iresnet("ir101", 512).eval()
        self.assertEqual(model.fc.in_features, 25088)
        # Downsample sits on conv2, matching the InsightFace IBasicBlock.
        self.assertEqual(model.layer1[0].conv1.stride, (1, 1))
        self.assertEqual(model.layer1[0].conv2.stride, (2, 2))
        self.assertEqual(model.layer1[1].conv2.stride, (1, 1))
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 112, 112))
        self.assertEqual(tuple(output.shape), (2, 512))

    def test_prefixed_checkpoint_loads_strictly(self):
        model = build_iresnet("ir18", 32)
        prefixed = {f"model.net.{key}": value.clone() for key, value in model.state_dict().items()}
        other = build_iresnet("ir18", 32)
        load_arcface_state(other, prefixed)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(other.state_dict()[key], value))

    def test_unknown_architecture_and_mismatched_keys_are_refused(self):
        with self.assertRaises(ValueError):
            build_iresnet("dinov2", 512)
        model = build_iresnet("ir18", 16)
        with self.assertRaises(ValueError):
            load_arcface_state(model, {"model.net.not_a_layer.weight": torch.zeros(1)})

    def test_vectors_resize_and_return_one_row(self):
        embedder = ArcFaceEmbedder()
        embedder._device = torch.device("cpu")
        embedder._model = build_iresnet("ir18", 8).eval()
        rows = embedder.vectors([Image.new("RGB", (40, 70), (10, 20, 30))])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].shape, (8,))


if __name__ == "__main__":
    unittest.main()
