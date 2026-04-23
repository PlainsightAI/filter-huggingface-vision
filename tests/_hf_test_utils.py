"""Shared test utilities for HuggingFace Hub error-handling tests."""
from unittest.mock import MagicMock


def make_hf_error(cls, message):
    """Build a huggingface_hub error instance with a minimal mock response
    attached, suitable for use in backend load-error tests."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    return cls(message, response=mock_response)
