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
    Both the processor-load and the model-load try blocks are covered.
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

    def test_object_detection_propagates_connection_error(self):
        from filter_huggingface_vision.backends.object_detection import ObjectDetectionBackend

        backend = ObjectDetectionBackend()
        cfg = SimpleNamespace(model_id="x", revision="main", device="cpu")
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=ConnectionError("hub unreachable"),
        ):
            with self.assertRaises(ConnectionError):
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
        # Processor-load failure must name AutoImageProcessor, not the model class.
        msg = str(ctx.exception)
        self.assertIn("not compatible", msg.lower())
        self.assertIn("AutoImageProcessor", msg)

    def test_object_detection_model_load_failure_names_model_class(self):
        from filter_huggingface_vision.backends.object_detection import ObjectDetectionBackend

        backend = ObjectDetectionBackend()
        cfg = SimpleNamespace(model_id="x", revision="main", device="cpu")
        # Processor succeeds; only the model load raises a compat error.
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            return_value=mock.MagicMock(),
        ), mock.patch(
            "transformers.AutoModelForObjectDetection.from_pretrained",
            side_effect=ValueError("not a supported model"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                backend.load(cfg)
        msg = str(ctx.exception)
        self.assertIn("not compatible", msg.lower())
        self.assertIn("AutoModelForObjectDetection", msg)
        self.assertNotIn("AutoImageProcessor", msg)

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

    def test_image_classification_propagates_memory_error(self):
        from filter_huggingface_vision.backends.image_classification import (
            ImageClassificationBackend,
        )

        backend = ImageClassificationBackend()
        cfg = SimpleNamespace(model_id="x", revision="main", device="cpu")
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=MemoryError("oom"),
        ):
            with self.assertRaises(MemoryError):
                backend.load(cfg)

    def test_image_classification_propagates_connection_error(self):
        from filter_huggingface_vision.backends.image_classification import (
            ImageClassificationBackend,
        )

        backend = ImageClassificationBackend()
        cfg = SimpleNamespace(model_id="x", revision="main", device="cpu")
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            side_effect=ConnectionError("hub unreachable"),
        ):
            with self.assertRaises(ConnectionError):
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
        msg = str(ctx.exception)
        self.assertIn("not compatible", msg.lower())
        self.assertIn("AutoImageProcessor", msg)

    def test_image_classification_model_load_failure_names_model_class(self):
        from filter_huggingface_vision.backends.image_classification import (
            ImageClassificationBackend,
        )

        backend = ImageClassificationBackend()
        cfg = SimpleNamespace(model_id="x", revision="main", device="cpu")
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            return_value=mock.MagicMock(),
        ), mock.patch(
            "transformers.AutoModelForImageClassification.from_pretrained",
            side_effect=ValueError("not a supported model"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                backend.load(cfg)
        msg = str(ctx.exception)
        self.assertIn("not compatible", msg.lower())
        self.assertIn("AutoModelForImageClassification", msg)
        self.assertNotIn("AutoImageProcessor", msg)


class TestEmbeddingBackendDoesNotSwallowInfraErrors(unittest.TestCase):
    """Fix #6 (extension): EmbeddingBackend._load_transformers must not retry OSError
    or RuntimeError across every Auto class — only compatibility-related exceptions
    (ValueError, TypeError, KeyError) should fall through to the next class.
    """

    def _cfg(self):
        return SimpleNamespace(
            model_id="x",
            revision="main",
            device="cpu",
            model_loader="transformers",
            exemplar_embeddings_path="",
            output_embeddings=True,
            output_distances=True,
        )

    def test_propagates_oserror_from_first_auto_class(self):
        from filter_huggingface_vision.backends.embedding import EmbeddingBackend

        backend = EmbeddingBackend()
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            return_value=mock.MagicMock(),
        ), mock.patch(
            "transformers.AutoModel.from_pretrained",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(OSError):
                backend.load(self._cfg())

    def test_propagates_runtime_error_from_first_auto_class(self):
        from filter_huggingface_vision.backends.embedding import EmbeddingBackend

        backend = EmbeddingBackend()
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            return_value=mock.MagicMock(),
        ), mock.patch(
            "transformers.AutoModel.from_pretrained",
            side_effect=RuntimeError("cuda oom"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                backend.load(self._cfg())
        # Must surface the original cause, not the generic "not compatible" wrapper.
        self.assertIn("cuda oom", str(ctx.exception).lower())

    def test_retries_on_value_error_across_auto_classes(self):
        from filter_huggingface_vision.backends.embedding import EmbeddingBackend

        backend = EmbeddingBackend()
        # First class fails with a compat error; subsequent class succeeds.
        fake_model = mock.MagicMock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model
        fake_model.named_children.return_value = []
        with mock.patch(
            "transformers.AutoImageProcessor.from_pretrained",
            return_value=mock.MagicMock(),
        ), mock.patch(
            "transformers.AutoModel.from_pretrained",
            side_effect=ValueError("incompat"),
        ), mock.patch(
            "transformers.AutoModelForImageClassification.from_pretrained",
            return_value=fake_model,
        ):
            backend.load(self._cfg())
        self.assertIs(backend._model, fake_model)


class TestGroundingDinoTextThresholdRangeCheck(unittest.TestCase):
    """Fix #4 [NIT]: text_threshold is now a declared config field with [0.0, 1.0] range."""

    def test_raises_on_out_of_range_text_threshold(self):
        from filter_huggingface_vision.backends.grounding_dino import GroundingDinoBackend

        backend = GroundingDinoBackend()
        cfg = SimpleNamespace(
            threshold=0.3,
            text_threshold=1.5,
            max_detections=100,
            text_labels=["dog"],
        )
        with self.assertRaises(ValueError) as ctx:
            backend.run(None, 100, 100, cfg)
        self.assertIn("text_threshold", str(ctx.exception))

    def test_text_threshold_declared_on_config(self):
        # text_threshold is now a declared, documented field (was an undocumented
        # dict lookup before). Defaults to None so the backend can fall back to
        # `threshold` when unset.
        from filter_huggingface_vision.filter import FilterHuggingfaceVisionConfig

        self.assertIn("text_threshold", FilterHuggingfaceVisionConfig.__annotations__)
        self.assertIsNone(FilterHuggingfaceVisionConfig.text_threshold)


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
