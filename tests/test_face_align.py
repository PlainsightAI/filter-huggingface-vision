"""Five-point alignment. No landmark-weight download."""

import unittest

import numpy as np
import torch
from PIL import Image

from filter_huggingface_vision.face_align import (
    ARCFACE_112,
    RetinaFace,
    generate_priors,
    match_landmarks,
    similarity_matrix,
    warp_face,
)


class SimilarityTests(unittest.TestCase):
    def test_similarity_recovers_scale_rotation_and_translation(self):
        theta = np.deg2rad(20.0)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        scale = 1.7
        translation = np.array([30.0, -12.0])
        source = ARCFACE_112.copy()
        target = (scale * (rotation @ source.T)).T + translation
        matrix = similarity_matrix(source, target)
        mapped = (matrix[:, :2] @ source.T).T + matrix[:, 2]
        np.testing.assert_allclose(mapped, target, atol=1e-6)

    def test_warp_lands_on_the_arcface_template(self):
        image = Image.new("RGB", (180, 160), (0, 0, 0))
        pixels = image.load()
        shifted = ARCFACE_112 + np.array([20.0, 15.0])
        for x, y in shifted:
            pixels[int(round(x)), int(round(y))] = (255, 255, 255)
        warped = warp_face(image, shifted)
        self.assertEqual(warped.size, (112, 112))
        matrix = similarity_matrix(shifted, ARCFACE_112)
        mapped = (matrix[:, :2] @ shifted.T).T + matrix[:, 2]
        np.testing.assert_allclose(mapped, ARCFACE_112, atol=1e-4)

    def test_collapsed_landmarks_are_refused(self):
        image = Image.new("RGB", (40, 40))
        self.assertIsNone(warp_face(image, np.zeros((5, 2))))


class MatchTests(unittest.TestCase):
    def test_center_inside_the_box_beats_an_outside_face(self):
        inside = (np.array([12.0, 12, 30, 30]), np.arange(10, dtype=float).reshape(5, 2), 0.4)
        outside = (np.array([40.0, 40, 90, 90]), np.ones((5, 2)), 0.99)
        points = match_landmarks((0, 0, 50, 50), [outside, inside])
        np.testing.assert_array_equal(points, inside[1])

    def test_iou_fallback_and_a_miss(self):
        face = (np.array([0.0, 0, 100, 50]), np.ones((5, 2)), 0.9)
        points = match_landmarks((0, 26, 100, 70), [face])
        np.testing.assert_array_equal(points, face[1])
        self.assertIsNone(match_landmarks((200, 200, 240, 240), [face]))


class RetinaFaceShapeTests(unittest.TestCase):
    def test_forward_matches_the_checkpoint_head_shapes(self):
        model = RetinaFace().eval()
        weight = model.LandmarkHead._convs[0].weight
        self.assertEqual(tuple(weight.shape), (20, 128, 1, 1))
        self.assertIn("body.layer4.0.conv1.weight", model.state_dict())
        with torch.inference_mode():
            boxes, scores, landmarks = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(boxes.shape[-1], 4)
        self.assertEqual(scores.shape[-1], 2)
        self.assertEqual(landmarks.shape[-1], 10)
        self.assertEqual(boxes.shape[1], landmarks.shape[1])
        self.assertEqual(boxes.shape[1], generate_priors(64, 64, "cpu").shape[0])


if __name__ == "__main__":
    unittest.main()
