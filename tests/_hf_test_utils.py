"""Shared test utilities for HuggingFace Hub error-handling tests."""
from unittest.mock import MagicMock


def make_hf_error(cls, message, status_code=404):
    """Build a huggingface_hub error instance with a minimal mock response
    attached, suitable for use in backend load-error tests.

    Pass `status_code` when a test needs to simulate something other than 404
    (e.g. 401 for gated access, 503 for transient Hub failures).
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    return cls(message, response=mock_response)
