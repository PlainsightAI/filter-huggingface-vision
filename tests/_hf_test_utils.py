"""Shared test utilities for HuggingFace Hub error-handling tests."""
from unittest.mock import MagicMock, patch


def make_hf_error(cls, message, status_code=404):
    """Build a huggingface_hub error instance with a minimal mock response
    attached, suitable for use in backend load-error tests.

    Pass `status_code` when a test needs to simulate something other than 404
    (e.g. 401 for gated access, 503 for transient Hub failures).
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    return cls(message, response=mock_response)


class HFLoadErrorTestsMixin:
    """Parameterized test suite for the shared `hf_load_error_handler`.

    Subclasses MUST override:
      backend_cls              — VisionBackend concrete class
      processor_patch_target   — dotted path to *Processor.from_pretrained
      model_patch_target       — dotted path to *Model.from_pretrained
      task_phrase              — human string passed to hf_load_error_handler
    """

    _CONFIG = {"model_id": "org/model", "revision": "abc123", "device": "cpu"}

    backend_cls = None
    processor_patch_target = None
    model_patch_target = None
    task_phrase = None

    def _load_backend(self):
        self.backend_cls().load(self._CONFIG)

    def _patch_processor(self, **kwargs):
        return patch(self.processor_patch_target, **kwargs)

    def _patch_model(self, **kwargs):
        return patch(self.model_patch_target, **kwargs)

    # --- ImportError branches (S4 allowlist) ---

    def test_timm_import_error_gives_actionable_message(self):
        with self._patch_processor(side_effect=ImportError("No module named 'timm'")):
            with self.assertRaises(ImportError) as ctx:
                self._load_backend()
        self.assertIn("timm", str(ctx.exception))
        self.assertIn("pip install timm", str(ctx.exception))

    def test_sentencepiece_import_error_gives_actionable_message(self):
        with self._patch_processor(side_effect=ImportError("No module named 'sentencepiece'")):
            with self.assertRaises(ImportError) as ctx:
                self._load_backend()
        self.assertIn("sentencepiece", str(ctx.exception))
        self.assertIn("pip install sentencepiece", str(ctx.exception))

    def test_unknown_import_error_is_reraised_unchanged(self):
        original = ImportError("No module named 'some_other_dep'")
        with self._patch_processor(side_effect=original):
            with self.assertRaises(ImportError) as ctx:
                self._load_backend()
        self.assertIs(ctx.exception, original)

    # --- HuggingFace Hub error branches ---

    def test_repository_not_found_error_message(self):
        from huggingface_hub.utils import RepositoryNotFoundError

        exc = make_hf_error(RepositoryNotFoundError, "org/model")
        with self._patch_processor(side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("org/model", msg)
        self.assertIn("not found", msg)

    def test_repository_not_found_on_model_download_gives_actionable_message(self):
        from huggingface_hub.utils import RepositoryNotFoundError

        exc = make_hf_error(RepositoryNotFoundError, "org/model")
        with self._patch_processor(return_value=MagicMock()):
            with self._patch_model(side_effect=exc):
                with self.assertRaises(RuntimeError) as ctx:
                    self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("org/model", msg)
        self.assertIn("not found", msg)
        self.assertIs(ctx.exception.__cause__, exc)

    def test_revision_not_found_error_message(self):
        from huggingface_hub.utils import RevisionNotFoundError

        exc = make_hf_error(RevisionNotFoundError, "abc123")
        with self._patch_processor(side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("abc123", msg)
        self.assertIn("Revision", msg)

    def test_gated_repo_error_message(self):
        from huggingface_hub.utils import GatedRepoError

        exc = make_hf_error(GatedRepoError, "org/model")
        with self._patch_processor(side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("license", msg)
        self.assertIn("HF_TOKEN", msg)

    def test_hf_hub_http_error_includes_repr(self):
        from huggingface_hub.utils import HfHubHTTPError

        exc = make_hf_error(HfHubHTTPError, "503 Service Unavailable", status_code=503)
        with self._patch_processor(side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("org/model", msg)
        self.assertIn("HuggingFace Hub", msg)
        self.assertIn("503 Service Unavailable", msg)

    def test_hf_hub_http_401_hints_at_hf_token(self):
        from huggingface_hub.utils import HfHubHTTPError

        exc = make_hf_error(HfHubHTTPError, "401 Unauthorized", status_code=401)
        with self._patch_processor(side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("HF_TOKEN", msg)
        self.assertIn("gated", msg)

    def test_local_entry_not_found_error_message(self):
        from huggingface_hub.utils import LocalEntryNotFoundError

        exc = LocalEntryNotFoundError("cache miss for org/model")
        with self._patch_processor(side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("not found in cache", msg)
        self.assertIn("org/model", msg)

    def test_entry_not_found_error_message(self):
        from huggingface_hub.utils import EntryNotFoundError

        exc = EntryNotFoundError("entry not found for org/model")
        with self._patch_processor(side_effect=exc):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("not found in cache", msg)
        self.assertIn("org/model", msg)

    # --- ValueError / config-parse branch ---

    def test_value_error_gives_incompatibility_message_with_repr(self):
        with self._patch_processor(side_effect=ValueError("unrecognized architecture")):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn(self.task_phrase, msg)
        self.assertIn("unrecognized architecture", msg)

    # --- Fallback branch ---

    def test_unexpected_exception_includes_repr(self):
        with self._patch_processor(side_effect=OSError("disk full")):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        msg = str(ctx.exception)
        self.assertIn("Unexpected", msg)
        self.assertIn("disk full", msg)

    # --- Chained cause is preserved ---

    def test_original_cause_is_chained(self):
        original = ValueError("root cause")
        with self._patch_processor(side_effect=original):
            with self.assertRaises(RuntimeError) as ctx:
                self._load_backend()
        self.assertIs(ctx.exception.__cause__, original)
