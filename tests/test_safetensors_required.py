"""Every model load requires the safetensors format.

Loading a `.bin`/`.pt` checkpoint unpickles it, which runs code from the file on
the inference host. This filter already refuses `trust_remote_code`; the weight
format is the other door into the same problem, and this suite holds it shut on
the argument of every model load rather than on a comment.

Processors are not covered on purpose: they carry no weights.
"""
import unittest
from unittest.mock import MagicMock, patch


_CONFIG = {
    "model_id": "org/model",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "device": "cpu",
    "text_labels": ["a cat"],
}


def _loaded_model_kwargs(backend_import, model_cls_path, processor_cls_path, config=None):
    """Load a backend with both HF calls mocked, and return the model call kwargs."""
    module_path, cls_name = backend_import
    module = __import__(module_path, fromlist=[cls_name])
    backend_cls = getattr(module, cls_name)

    model = MagicMock()
    model.to.return_value = model
    with patch(processor_cls_path, return_value=MagicMock()), \
         patch(model_cls_path, return_value=model) as model_fp:
        backend_cls().load(config or _CONFIG)
    return model_fp.call_args.kwargs


class TestSafetensorsRequired(unittest.TestCase):
    def test_object_detection(self):
        kwargs = _loaded_model_kwargs(
            ("filter_huggingface_vision.backends.object_detection", "ObjectDetectionBackend"),
            "transformers.AutoModelForObjectDetection.from_pretrained",
            "transformers.AutoImageProcessor.from_pretrained",
        )
        self.assertIs(kwargs.get("use_safetensors"), True)

    def test_image_classification(self):
        kwargs = _loaded_model_kwargs(
            ("filter_huggingface_vision.backends.image_classification", "ImageClassificationBackend"),
            "transformers.AutoModelForImageClassification.from_pretrained",
            "transformers.AutoImageProcessor.from_pretrained",
        )
        self.assertIs(kwargs.get("use_safetensors"), True)

    def test_owlvit(self):
        kwargs = _loaded_model_kwargs(
            ("filter_huggingface_vision.backends.owlvit", "OwlVitBackend"),
            "transformers.AutoModelForZeroShotObjectDetection.from_pretrained",
            "transformers.AutoProcessor.from_pretrained",
        )
        self.assertIs(kwargs.get("use_safetensors"), True)

    def test_grounding_dino(self):
        kwargs = _loaded_model_kwargs(
            ("filter_huggingface_vision.backends.grounding_dino", "GroundingDinoBackend"),
            "transformers.AutoModelForZeroShotObjectDetection.from_pretrained",
            "transformers.AutoProcessor.from_pretrained",
        )
        self.assertIs(kwargs.get("use_safetensors"), True)


class TestPickleOnlyRepoIsRefusedWithAnActionableMessage(unittest.TestCase):
    """transformers reports the refusal as an OSError naming safetensors.

    The handler lets OSError propagate by policy, so this case has to be
    recognised explicitly — otherwise the operator gets a bare "does not appear
    to have a file named model.safetensors" with no indication that the filter
    asked for it on purpose.
    """

    def _load(self, exc):
        from filter_huggingface_vision.backends.object_detection import (
            ObjectDetectionBackend,
        )

        with patch("transformers.AutoImageProcessor.from_pretrained", return_value=MagicMock()), \
             patch("transformers.AutoModelForObjectDetection.from_pretrained", side_effect=exc):
            ObjectDetectionBackend().load(_CONFIG)

    def test_missing_safetensors_names_the_refusal(self):
        exc = OSError(
            "org/model does not appear to have a file named model.safetensors."
        )
        with self.assertRaises(RuntimeError) as ctx:
            self._load(exc)
        msg = str(ctx.exception)
        self.assertIn("safetensors", msg)
        self.assertIn("org/model", msg)
        self.assertIn("refuses pickle checkpoints", msg)

    def test_other_os_errors_still_propagate_unchanged(self):
        """The handler's policy: infrastructure failures keep their own type, so
        callers and retry logic still see an OSError."""
        exc = OSError("No space left on device")
        with self.assertRaises(OSError) as ctx:
            self._load(exc)
        self.assertNotIsInstance(ctx.exception, RuntimeError)
        self.assertIn("No space left", str(ctx.exception))
