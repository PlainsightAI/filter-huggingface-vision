"""Tests for embedding extraction backend (no model download)."""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch
import torch.nn as nn

from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)
from filter_huggingface_vision.backends.embedding import (
    EmbeddingBackend,
    _find_penultimate_module,
    _pool_embedding,
)


# ── helpers ────────────────────────────────────────────────────────────


class TestFindPenultimateModule(unittest.TestCase):
    """Unit tests for _find_penultimate_module."""

    def test_skips_classifier_head(self):
        model = nn.Sequential()
        model.add_module("features", nn.Linear(10, 10))
        model.add_module("classifier", nn.Linear(10, 2))
        result = _find_penultimate_module(model)
        self.assertIsInstance(result, nn.Linear)
        self.assertEqual(result.out_features, 10)

    def test_skips_head_named_head(self):
        model = nn.Module()
        model.backbone = nn.Linear(10, 10)
        model.head = nn.Linear(10, 5)
        result = _find_penultimate_module(model)
        self.assertEqual(result.out_features, 10)

    def test_returns_last_child_when_no_head(self):
        model = nn.Sequential()
        model.add_module("layer1", nn.Linear(10, 20))
        model.add_module("layer2", nn.Linear(20, 30))
        result = _find_penultimate_module(model)
        self.assertEqual(result.out_features, 30)


class TestPoolEmbedding(unittest.TestCase):
    """Unit tests for _pool_embedding."""

    def test_2d_input(self):
        t = torch.randn(1, 384)
        result = _pool_embedding(t)
        self.assertEqual(result.shape, (384,))

    def test_3d_input_takes_cls_token(self):
        t = torch.randn(1, 197, 768)
        result = _pool_embedding(t)
        self.assertEqual(result.shape, (768,))
        torch.testing.assert_close(result, t[0, 0])

    def test_4d_channels_first(self):
        # (B, D, H, W) where D > W → channels-first
        t = torch.randn(1, 768, 7, 7)
        result = _pool_embedding(t)
        self.assertEqual(result.shape, (768,))

    def test_4d_channels_last(self):
        # (B, H, W, D) where dim1 < dim3 → channels-last
        t = torch.randn(1, 7, 7, 768)
        result = _pool_embedding(t)
        self.assertEqual(result.shape, (768,))


# ── config ─────────────────────────────────────────────────────────────


class TestEmbeddingConfig(unittest.TestCase):
    """Config validation for embedding detection type."""

    def test_normalize_config_accepts_embedding(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="facebook/dinov2-small",
                revision="main",
                detection_type="embedding",
            )
        )
        self.assertEqual(config.detection_type, "embedding")

    def test_embedding_does_not_require_threshold(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="facebook/dinov2-small",
                revision="main",
                detection_type="embedding",
                threshold=0.0,
            )
        )
        self.assertEqual(config.detection_type, "embedding")

    def test_embedding_does_not_require_text_labels(self):
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(
                id="test",
                sources="",
                outputs="",
                model_id="facebook/dinov2-small",
                revision="main",
                detection_type="embedding",
            )
        )
        self.assertIsNone(config.text_labels)


# ── filter process ─────────────────────────────────────────────────────


class TestEmbeddingFilter(unittest.TestCase):
    """Filter process() with embedding backend (mock, no model download)."""

    def _make_mock_backend(self, run_return):
        class MockEmbeddingBackend:
            def load(self, config):
                pass

            def run(self, image_pil, width, height, config):
                return run_return

            def shutdown(self):
                pass

        return MockEmbeddingBackend

    def _setup_filter(self, mock_cls, **config_overrides):
        from PIL import Image

        defaults = dict(
            id="test",
            sources="",
            outputs="",
            model_id="facebook/dinov2-small",
            revision="main",
            detection_type="embedding",
        )
        defaults.update(config_overrides)
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(**defaults)
        )

        with patch(
            "filter_huggingface_vision.filter.get_backend",
            side_effect=lambda dt: mock_cls,
        ):
            f = FilterHuggingfaceVision(config)
            f.setup(config)
        return f, config

    def test_process_attaches_embedding_to_frame_data(self):
        from PIL import Image

        fake = np.random.randn(384).tolist()
        mock_cls = self._make_mock_backend({"embeddings": {"embedding": fake}})
        f, _ = self._setup_filter(mock_cls)

        pil = Image.new("RGB", (224, 224), color="red")
        frame = type("Frame", (), {"data": {"main": pil}})()
        try:
            f.process({"main": frame})
        finally:
            f.shutdown()

        self.assertEqual(frame.data["embedding"], fake)
        self.assertEqual(frame.data["meta"]["detection_type"], "embedding")
        self.assertEqual(frame.data["meta"]["task"], "embedding")

    def test_process_attaches_min_distance_only(self):
        from PIL import Image

        fake = np.random.randn(384).tolist()
        mock_cls = self._make_mock_backend(
            {"embeddings": {"embedding": fake, "min_exemplar_distance": 0.42}}
        )
        f, _ = self._setup_filter(mock_cls)

        pil = Image.new("RGB", (224, 224), color="blue")
        frame = type("Frame", (), {"data": {"main": pil}})()
        try:
            f.process({"main": frame})
        finally:
            f.shutdown()

        self.assertAlmostEqual(frame.data["min_exemplar_distance"], 0.42)
        self.assertNotIn("exemplar_distances", frame.data)
        self.assertNotIn("mean_exemplar_distance", frame.data)

    def test_process_preserves_upstream_meta(self):
        from PIL import Image

        mock_cls = self._make_mock_backend({"embeddings": {"embedding": [0.1]}})
        f, _ = self._setup_filter(mock_cls)

        pil = Image.new("RGB", (224, 224), color="green")
        frame = type(
            "Frame", (), {"data": {"main": pil, "meta": {"id": 42, "ts": 1234}}}
        )()
        try:
            f.process({"main": frame})
        finally:
            f.shutdown()

        self.assertEqual(frame.data["meta"]["id"], 42)
        self.assertEqual(frame.data["meta"]["ts"], 1234)
        self.assertIn("embedding", frame.data)


# ── backend unit tests ─────────────────────────────────────────────────


class TestEmbeddingBackendUnit(unittest.TestCase):
    """Unit tests for the EmbeddingBackend class."""

    def test_load_exemplars_with_embeddings_key(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = np.random.randn(5, 384).astype(np.float32)
            path = f"{tmpdir}/exemplars.npz"
            np.savez(path, embeddings=embeddings)

            backend = EmbeddingBackend.__new__(EmbeddingBackend)
            loaded = backend._load_exemplars(path)

            self.assertEqual(loaded.shape, (5, 384))
            np.testing.assert_array_equal(loaded, embeddings)

    def test_load_exemplars_file_not_found(self):
        backend = EmbeddingBackend.__new__(EmbeddingBackend)
        with self.assertRaises(FileNotFoundError):
            backend._load_exemplars("/nonexistent/path.npz")

    def test_hook_fn_captures_tensor(self):
        backend = EmbeddingBackend.__new__(EmbeddingBackend)
        backend._hooked_output = {}
        t = torch.randn(1, 384)
        backend._hook_fn(None, None, t)
        torch.testing.assert_close(backend._hooked_output["features"], t)

    def test_hook_fn_captures_from_tuple(self):
        backend = EmbeddingBackend.__new__(EmbeddingBackend)
        backend._hooked_output = {}
        t = torch.randn(1, 384)
        backend._hook_fn(None, None, (t, "other"))
        torch.testing.assert_close(backend._hooked_output["features"], t)

    def test_run_only_emits_min_distance(self):
        """Verify that run() emits min_exemplar_distance but not mean or full array."""
        import tempfile

        backend = EmbeddingBackend.__new__(EmbeddingBackend)
        backend._device = torch.device("cpu")
        backend._model_loader = "transformers"
        backend._uses_hook = False
        backend._hooked_output = {}
        backend._output_embeddings = True
        backend._output_distances = True

        # Set up exemplars
        exemplars = np.random.randn(10, 4).astype(np.float32)
        backend._exemplar_embeddings = exemplars

        # Mock the model + processor
        fake_hidden = torch.randn(1, 1, 4)
        mock_output = MagicMock()
        mock_output.last_hidden_state = fake_hidden
        mock_output.pooler_output = None
        backend._model = MagicMock(return_value=mock_output)
        backend._processor = MagicMock(
            return_value={"pixel_values": torch.randn(1, 3, 224, 224)}
        )

        from PIL import Image

        img = Image.new("RGB", (224, 224))
        result = backend.run(img, 224, 224, {})

        emb = result["embeddings"]
        self.assertIn("embedding", emb)
        self.assertIn("min_exemplar_distance", emb)
        self.assertNotIn("mean_exemplar_distance", emb)
        self.assertNotIn("exemplar_distances", emb)


class TestPoolEmbeddingEdgeCases(unittest.TestCase):
    """Additional _pool_embedding tests for edge cases."""

    def test_5d_tensor_fallback(self):
        """5D+ tensors should flatten to 1D via the fallback path."""
        t = torch.randn(1, 2, 3, 4, 5)
        result = _pool_embedding(t)
        self.assertEqual(result.dim(), 1)
        self.assertEqual(result.shape[0], 2 * 3 * 4 * 5)

    def test_4d_ambiguous_shape_warns(self):
        """Square-map tensor (dim1 == dim3) should emit a warning."""
        t = torch.randn(1, 64, 64, 64)
        with self.assertLogs(
            "filter_huggingface_vision.backends.embedding", level="WARNING"
        ) as cm:
            result = _pool_embedding(t)
        self.assertTrue(any("Ambiguous" in msg for msg in cm.output))
        # Should still produce a 1D result
        self.assertEqual(result.dim(), 1)


class TestEmbeddingBackendShutdown(unittest.TestCase):
    """Tests for shutdown() cleanup."""

    def test_shutdown_removes_hook_and_clears_state(self):
        backend = EmbeddingBackend.__new__(EmbeddingBackend)
        backend._processor = MagicMock()
        backend._model = MagicMock()
        backend._exemplar_embeddings = np.zeros((5, 384))
        backend._hooked_output = {"features": torch.randn(1, 384)}

        mock_handle = MagicMock()
        backend._hook_handle = mock_handle

        backend.shutdown()

        mock_handle.remove.assert_called_once()
        self.assertIsNone(backend._hook_handle)
        self.assertIsNone(backend._processor)
        self.assertIsNone(backend._model)
        self.assertIsNone(backend._exemplar_embeddings)
        self.assertEqual(backend._hooked_output, {})

    def test_shutdown_no_hook(self):
        """shutdown() works when no hook was installed."""
        backend = EmbeddingBackend.__new__(EmbeddingBackend)
        backend._processor = MagicMock()
        backend._model = MagicMock()
        backend._exemplar_embeddings = None
        backend._hooked_output = {}
        backend._hook_handle = None

        backend.shutdown()  # should not raise

        self.assertIsNone(backend._processor)
        self.assertIsNone(backend._model)


class TestEmbeddingBackendRunEdgeCases(unittest.TestCase):
    """Tests for run() edge cases."""

    def _make_backend(self, **kwargs):
        backend = EmbeddingBackend.__new__(EmbeddingBackend)
        backend._device = torch.device("cpu")
        backend._model_loader = kwargs.get("model_loader", "transformers")
        backend._uses_hook = kwargs.get("uses_hook", False)
        backend._hooked_output = {}
        backend._output_embeddings = kwargs.get("output_embeddings", True)
        backend._output_distances = kwargs.get("output_distances", True)
        backend._exemplar_embeddings = kwargs.get("exemplar_embeddings", None)

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 1, 4)
        mock_output.pooler_output = None
        backend._model = MagicMock(return_value=mock_output)
        backend._processor = MagicMock(
            return_value={"pixel_values": torch.randn(1, 3, 224, 224)}
        )
        return backend

    def test_output_embeddings_false_excludes_embedding(self):
        """When output_embeddings=False, result should not contain 'embedding' key."""
        from PIL import Image

        backend = self._make_backend(output_embeddings=False, exemplar_embeddings=None)
        result = backend.run(Image.new("RGB", (224, 224)), 224, 224, {})

        self.assertNotIn("embedding", result["embeddings"])

    def test_output_embeddings_false_no_exemplars_empty_dict(self):
        """output_embeddings=False + no exemplars = empty embeddings dict."""
        from PIL import Image

        backend = self._make_backend(
            output_embeddings=False, output_distances=True, exemplar_embeddings=None
        )
        result = backend.run(Image.new("RGB", (224, 224)), 224, 224, {})

        self.assertEqual(result["embeddings"], {})

    def test_run_value_error_when_no_features(self):
        """run() raises ValueError when model output has no extractable features."""
        from PIL import Image

        backend = self._make_backend()
        # Make model return an object with no usable attributes
        mock_output = MagicMock(spec=[])
        mock_output.last_hidden_state = None
        mock_output.pooler_output = None
        del mock_output.last_hidden_state
        del mock_output.pooler_output

        backend._model = MagicMock(return_value=mock_output)

        with self.assertRaises(ValueError):
            backend.run(Image.new("RGB", (224, 224)), 224, 224, {})

    def test_timm_loading_path(self):
        """run() with timm model_loader uses the processor as a transform."""
        from PIL import Image

        backend = EmbeddingBackend.__new__(EmbeddingBackend)
        backend._device = torch.device("cpu")
        backend._model_loader = "timm"
        backend._uses_hook = False
        backend._hooked_output = {}
        backend._output_embeddings = True
        backend._output_distances = False
        backend._exemplar_embeddings = None

        # timm processor is a transform callable that returns a tensor
        fake_tensor = torch.randn(3, 224, 224)
        backend._processor = MagicMock(return_value=fake_tensor)
        # timm model returns a 2D tensor directly
        backend._model = MagicMock(return_value=torch.randn(1, 512))

        result = backend.run(Image.new("RGB", (224, 224)), 224, 224, {})

        self.assertIn("embedding", result["embeddings"])
        self.assertEqual(len(result["embeddings"]["embedding"]), 512)
        # Verify processor was called with the PIL image
        backend._processor.assert_called_once()


class TestEmbeddingFilterVisualizationWarning(unittest.TestCase):
    """Test that draw_visualization with embedding emits a warning."""

    def test_draw_visualization_with_embedding_warns(self):
        from PIL import Image

        fake = [0.1, 0.2, 0.3]

        class MockBackend:
            def load(self, config):
                pass

            def run(self, image_pil, width, height, config):
                return {"embeddings": {"embedding": fake}}

            def shutdown(self):
                pass

        defaults = dict(
            id="test",
            sources="",
            outputs="",
            model_id="facebook/dinov2-small",
            revision="main",
            detection_type="embedding",
            draw_visualization=True,
        )
        config = FilterHuggingfaceVision.normalize_config(
            FilterHuggingfaceVisionConfig(**defaults)
        )

        with patch(
            "filter_huggingface_vision.filter.get_backend",
            side_effect=lambda dt: MockBackend,
        ):
            f = FilterHuggingfaceVision(config)
            with self.assertLogs(
                "filter_huggingface_vision.filter", level="WARNING"
            ) as cm:
                f.setup(config)

        self.assertTrue(any("draw_visualization" in msg for msg in cm.output))
        f.shutdown()


if __name__ == "__main__":
    unittest.main()
