"""Shared HuggingFace Hub load-error handler for vision backends."""
from contextlib import contextmanager

from huggingface_hub import errors as _hf_errors


@contextmanager
def hf_load_error_handler(model_id: str, revision: str, task: str):
    """Wraps AutoImageProcessor + AutoModelFor*.from_pretrained calls with
    targeted, actionable error messages.

    GatedRepoError must be caught before RepositoryNotFoundError because it
    is a subclass of it in huggingface_hub >= 0.22 / 1.x.
    """
    try:
        yield
    except ImportError as e:
        if "timm" in str(e).lower():
            raise ImportError(
                f"This model (e.g. {model_id}) requires the timm library. "
                "Install it with: pip install timm"
            ) from e
        raise
    except _hf_errors.GatedRepoError as e:
        raise RuntimeError(
            f"Access to '{model_id}' requires accepting its license on the Hub. "
            f"Accept it with an authenticated token. Detail: {repr(e)}"
        ) from e
    except _hf_errors.RepositoryNotFoundError as e:
        raise RuntimeError(
            f"HuggingFace repository '{model_id}' not found. "
            f"Check the model ID or your HF_TOKEN if the repo is gated. "
            f"Detail: {repr(e)}"
        ) from e
    except _hf_errors.RevisionNotFoundError as e:
        raise RuntimeError(
            f"Revision '{revision}' not found in '{model_id}'. "
            f"Check the revision (commit SHA, tag, or branch). "
            f"Detail: {repr(e)}"
        ) from e
    except _hf_errors.HfHubHTTPError as e:
        raise RuntimeError(
            f"Could not download '{model_id}@{revision}' from HuggingFace Hub: {repr(e)}"
        ) from e
    except ValueError as e:
        raise RuntimeError(
            f"Model '{model_id}' could not be loaded for {task}. "
            f"Check that the model's architecture exposes the required head: {repr(e)}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected failure loading '{model_id}@{revision}' for {task}: {repr(e)}"
        ) from e
