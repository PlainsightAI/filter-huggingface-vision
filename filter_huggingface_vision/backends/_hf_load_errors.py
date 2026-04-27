"""Shared HuggingFace Hub load-error handler for vision backends."""
from contextlib import contextmanager

# Import from huggingface_hub.utils — this path has been stable since hub ~0.16
# and is available at the declared floor (>=0.23), unlike huggingface_hub.errors
# which was introduced later. Both modules expose the same classes.
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

# Allowlist of known optional dependencies that produce ImportError.
# Maps lower-cased dep name → install hint shown to the operator.
_MISSING_DEP_HINTS = {
    "timm": "pip install timm (needed by DETR / Conditional DETR)",
    "sentencepiece": "pip install sentencepiece",
    "detectron2": "pip install detectron2 — see upstream install guide",
    "torchvision": "pip install torchvision",
}


@contextmanager
def hf_load_error_handler(model_id: str, revision: str, task: str):
    """Wraps AutoImageProcessor + AutoModelFor*.from_pretrained calls with
    targeted, actionable error messages.

    Catch order matters:
    - GatedRepoError before RepositoryNotFoundError (it is a subclass).
    - RepositoryNotFoundError and RevisionNotFoundError before HfHubHTTPError
      (both are subclasses).
    - (LocalEntryNotFoundError, EntryNotFoundError) before HfHubHTTPError
      and before the `except ValueError` fallback: on huggingface-hub 0.23
      (our declared floor, with the `requests` backbone) both ARE subclasses
      of HfHubHTTPError, and LocalEntryNotFoundError is additionally a
      subclass of ValueError, so any reorder would silently reroute them to
      a less helpful branch. (Hub >=1.0 flattened the hierarchy; keeping the
      order preserves correct behavior on both eras.)
    """
    try:
        yield
    except ImportError as e:
        msg = str(e).lower()
        for dep, hint in _MISSING_DEP_HINTS.items():
            if dep in msg:
                raise ImportError(
                    f"Model '{model_id}' requires '{dep}': {hint}. "
                    f"Original: {repr(e)}"
                ) from e
        raise
    except GatedRepoError as e:
        raise RuntimeError(
            f"Access to '{model_id}' requires accepting its license on the Hub. "
            f"Accept it with an authenticated token "
            f"(set HF_TOKEN environment variable). Detail: {repr(e)}"
        ) from e
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"HuggingFace repository '{model_id}' not found. "
            f"Check the model ID or your HF_TOKEN if the repo is gated. "
            f"Detail: {repr(e)}"
        ) from e
    except RevisionNotFoundError as e:
        raise RuntimeError(
            f"Revision '{revision}' not found in '{model_id}'. "
            f"Check the revision (commit SHA, tag, or branch). "
            f"Detail: {repr(e)}"
        ) from e
    # Catch order is load-bearing here: at hub 0.23 (our declared floor) these
    # are subclasses of HfHubHTTPError, and LocalEntryNotFoundError is also a
    # subclass of ValueError; do not reorder this branch below either of them.
    except (LocalEntryNotFoundError, EntryNotFoundError) as e:
        raise RuntimeError(
            f"Model entry for '{model_id}@{revision}' could not be resolved: the "
            f"repository may be missing the expected file, or the local cache may "
            f"be incomplete. If running offline, pre-download the model with "
            f"`huggingface-cli download {model_id} --revision {revision}` before "
            f"starting the filter; if concurrent workers share a cache, retry "
            f"after the first worker completes. Detail: {repr(e)}"
        ) from e
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 401:
            raise RuntimeError(
                f"Unauthorized loading '{model_id}@{revision}' — the model may be "
                f"gated or require an HF_TOKEN. Set HF_TOKEN and retry. Detail: {repr(e)}"
            ) from e
        raise RuntimeError(
            f"Could not download '{model_id}@{revision}' from HuggingFace Hub: {repr(e)}"
        ) from e
    except ValueError as e:
        raise RuntimeError(
            f"Model '{model_id}@{revision}' could not be loaded for {task}. "
            f"Detail: {repr(e)}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected failure loading '{model_id}@{revision}' for {task}: {repr(e)}"
        ) from e
