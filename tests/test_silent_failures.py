"""Tests for PLAT-889: silent failure patterns in filter-huggingface-vision.

Each test asserts that an error path either raises or logs, instead of
silently swallowing the failure and returning empty/wrong data.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock


class TestImageFromFramePropagatesImportError(unittest.TestCase):
    """Fix #1: filter.py:_image_from_frame must not silently swallow ImportError."""

    def test_propagates_when_cv2_missing(self):
        from filter_huggingface_vision.filter import _image_from_frame

        with mock.patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaises(ImportError):
                _image_from_frame(SimpleNamespace(has_image=False, data=None), "main")


class TestCreateVisualizationPropagatesImportError(unittest.TestCase):
    """Fix #2: filter.py:_create_visualization must not silently swallow ImportError."""

    def test_propagates_when_cv2_missing(self):
        from filter_huggingface_vision.filter import _create_visualization

        with mock.patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaises(ImportError):
                _create_visualization(None, {"detections": []})


class TestZeroShotBackendsLogOnEmptyTextLabels(unittest.TestCase):
    """Fix #3: OWL-ViT and Grounding DINO must log a warning when text_labels is empty/invalid."""

    def _config(self, text_labels):
        return SimpleNamespace(text_labels=text_labels, threshold=0.1, max_detections=100)

    def test_owlvit_logs_warning_on_empty_text_labels(self):
        from filter_huggingface_vision.backends.owlvit import OwlVitBackend

        backend = OwlVitBackend()
        with self.assertLogs("filter_huggingface_vision.backends.owlvit", level="WARNING") as cm:
            out = backend.run(None, 100, 100, self._config([]))
        self.assertEqual(out, [])
        self.assertTrue(any("text_labels" in m for m in cm.output))

    def test_owlvit_logs_warning_on_invalid_text_labels_type(self):
        from filter_huggingface_vision.backends.owlvit import OwlVitBackend

        backend = OwlVitBackend()
        with self.assertLogs("filter_huggingface_vision.backends.owlvit", level="WARNING") as cm:
            out = backend.run(None, 100, 100, self._config("not-a-list"))
        self.assertEqual(out, [])
        self.assertTrue(any("text_labels" in m for m in cm.output))

    def test_grounding_dino_logs_warning_on_empty_text_labels(self):
        from filter_huggingface_vision.backends.grounding_dino import GroundingDinoBackend

        backend = GroundingDinoBackend()
        with self.assertLogs("filter_huggingface_vision.backends.grounding_dino", level="WARNING") as cm:
            out = backend.run(None, 100, 100, self._config([]))
        self.assertEqual(out, [])
        self.assertTrue(any("text_labels" in m for m in cm.output))


class TestApplyMetaRaisesOnMissingDetectionType(unittest.TestCase):
    """Fix #4: _apply_meta must use raise ValueError (not assert, which is stripped under -O)."""

    def test_raises_value_error_when_detection_type_missing(self):
        from filter_huggingface_vision.filter import _apply_meta

        meta = {}
        payload = {"task": "object-detection"}  # no detection_type
        with self.assertRaises(ValueError) as ctx:
            _apply_meta(meta, payload, SimpleNamespace(id="t"))
        self.assertIn("detection_type", str(ctx.exception))


class TestResolveDeviceLogsCudaFallback(unittest.TestCase):
    """Fix #5: resolve_device must log a WARNING when CUDA is requested but unavailable."""

    def test_logs_warning_when_cuda_string_requested_but_unavailable(self):
        from filter_huggingface_vision import utils

        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertLogs("filter_huggingface_vision.utils", level="WARNING") as cm:
                dev = utils.resolve_device("cuda:0")
        self.assertEqual(dev.type, "cpu")
        self.assertTrue(any("cuda" in m.lower() for m in cm.output))

    def test_logs_warning_when_cuda_int_requested_but_unavailable(self):
        from filter_huggingface_vision import utils

        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertLogs("filter_huggingface_vision.utils", level="WARNING") as cm:
                dev = utils.resolve_device(0)
        self.assertEqual(dev.type, "cpu")
        self.assertTrue(any("cuda" in m.lower() for m in cm.output))


class TestBackendsDontMisattributeInfraErrors(unittest.TestCase):
    """Fix #6: object_detection and image_classification backends must not relabel
    infra errors (OSError, MemoryError, ConnectionError) as 'model not compatible'.
    Only narrow compatibility-related exceptions get the RuntimeError treatment.
    """

    def test_object_detection_propagates_oserror(self):
        from filter_huggingface_vision.backends.object_detection import ObjectDetectionBackend

        backend = ObjectDetectionBackend()
        cfg = SimpleNamespace(model_id="x", revision="main", device="cpu")
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(OSError):
                backend.load(cfg)

    def test_object_detection_propagates_memory_error(self):
        from filter_huggingface_vision.backends.object_detection import ObjectDetectionBackend

        backend = ObjectDetectionBackend()
        cfg = SimpleNamespace(model_id="x", revision="main", device="cpu")
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=MemoryError("oom"),
        ):
            with self.assertRaises(MemoryError):
                backend.load(cfg)

    def test_object_detection_wraps_value_error_as_runtime_error(self):
        from filter_huggingface_vision.backends.object_detection import ObjectDetectionBackend

        backend = ObjectDetectionBackend()
        cfg = SimpleNamespace(model_id="x", revision="main", device="cpu")
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=ValueError("not a supported config"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                backend.load(cfg)
        self.assertIn("not compatible", str(ctx.exception).lower())

    def test_image_classification_propagates_oserror(self):
        from filter_huggingface_vision.backends.image_classification import (
            ImageClassificationBackend,
        )

        backend = ImageClassificationBackend()
        cfg = SimpleNamespace(model_id="x", revision="main", device="cpu")
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(OSError):
                backend.load(cfg)

    def test_image_classification_wraps_value_error_as_runtime_error(self):
        from filter_huggingface_vision.backends.image_classification import (
            ImageClassificationBackend,
        )

        backend = ImageClassificationBackend()
        cfg = SimpleNamespace(model_id="x", revision="main", device="cpu")
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=ValueError("not a supported config"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                backend.load(cfg)
        self.assertIn("not compatible", str(ctx.exception).lower())


class TestCvtColorOnlyCatchesCvError(unittest.TestCase):
    """Fix #7: filter.py:_image_from_frame must only catch cv2.error around cv2.cvtColor
    (and log a warning). MemoryError / KeyboardInterrupt must propagate.
    """

    def _frame_with_array(self, arr):
        # frame.data is the ndarray itself, so the fallback path `raw = data` is taken
        # (ndarrays lack a `.get` method).
        return SimpleNamespace(has_image=False, data=arr)

    def test_logs_warning_on_cv2_error_and_falls_back_to_input_array(self):
        import cv2
        import numpy as np

        from filter_huggingface_vision.filter import _image_from_frame

        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        frame = self._frame_with_array(arr)
        with mock.patch("cv2.cvtColor", side_effect=cv2.error("bad input")):
            with self.assertLogs("filter_huggingface_vision.filter", level="WARNING") as cm:
                img, w, h = _image_from_frame(frame, "main")
        self.assertIsNotNone(img)
        self.assertEqual((w, h), (4, 4))
        self.assertTrue(any("cvtcolor" in m.lower() or "cv2" in m.lower() for m in cm.output))

    def test_propagates_unrelated_exceptions(self):
        import numpy as np

        from filter_huggingface_vision.filter import _image_from_frame

        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        frame = self._frame_with_array(arr)
        with mock.patch("cv2.cvtColor", side_effect=MemoryError("oom")):
            with self.assertRaises(MemoryError):
                _image_from_frame(frame, "main")


class TestProcessLogsWhenBackendUninitialized(unittest.TestCase):
    """Fix #8: process() must log a WARNING when frames pass through without an initialized backend."""

    def test_logs_warning_when_backend_is_none_and_frames_present(self):
        from filter_huggingface_vision.filter import FilterHuggingfaceVision

        filt = FilterHuggingfaceVision.__new__(FilterHuggingfaceVision)
        filt._backend = None
        frames = {"main": SimpleNamespace(has_image=False, data={})}

        with self.assertLogs("filter_huggingface_vision.filter", level="WARNING") as cm:
            out = filt.process(frames)
        self.assertIs(out, frames)
        self.assertTrue(any("backend" in m.lower() for m in cm.output))

    def test_no_log_when_no_frames(self):
        from filter_huggingface_vision.filter import FilterHuggingfaceVision

        filt = FilterHuggingfaceVision.__new__(FilterHuggingfaceVision)
        filt._backend = None
        with self.assertNoLogs("filter_huggingface_vision.filter", level="WARNING"):
            out = filt.process({})
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
