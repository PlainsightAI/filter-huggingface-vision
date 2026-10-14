"""Embedding extraction backend: extract penultimate-layer feature vectors from any vision model.

Uses PyTorch forward hooks to capture the last representation before the output
head, making it model-agnostic.  Works with classification models, detection
models, or pure feature extractors — anything that produces spatial or sequence
features before a task-specific head.

Model loading:
- model_loader="transformers": any HuggingFace model loadable via AutoModel or a
  task-specific AutoModelFor* class (AutoModelForImageClassification, etc.)
- model_loader="timm": any timm model (uses num_classes=0 to strip the head)
"""

import logging

import fsspec
import numpy as np
import torch
import torch.nn as nn

from filter_huggingface_vision.utils import get_config_value, resolve_device

from .base import VisionBackend

logger = logging.getLogger(__name__)

# HuggingFace task-specific Auto classes to try, in order of preference.
# AutoModel (base) is tried first; if it fails we fall back to headed variants
# and use a forward hook to grab penultimate features.
_HF_AUTO_CLASSES = [
    "AutoModel",
    "AutoModelForImageClassification",
    "AutoModelForObjectDetection",
    "AutoModelForZeroShotObjectDetection",
    "AutoModelForSemanticSegmentation",
    "AutoModelForDepthEstimation",
]

# Module names commonly used for the output head in HF and timm models.
_HEAD_NAMES = frozenset({
    "classifier", "head", "fc", "lm_head", "score",
    "class_head", "bbox_predictor", "mask_predictor",
    "decode_head", "segmentation_head",
})


def _find_penultimate_module(model: nn.Module) -> nn.Module:
    """Return the last module before the output head.

    Strategy:
    1. Walk top-level children; for each, check whether its name or any of its
       descendants' names match a known head name.
    2. Return the last top-level child that is NOT (and does not contain) a head.
    3. If no head is found (e.g. AutoModel already stripped it), return the
       last child — its output is the final representation.
    """
    children = list(model.named_children())
    if not children:
        return model

    def _is_or_contains_head(name: str, module: nn.Module) -> bool:
        if name in _HEAD_NAMES:
            return True
        for sub_name, _ in module.named_modules():
            if sub_name and sub_name.rsplit(".", 1)[-1] in _HEAD_NAMES:
                return True
        return False

    last_non_head = None
    for name, module in children:
        if not _is_or_contains_head(name, module):
            last_non_head = module

    return last_non_head if last_non_head is not None else children[-1][1]


def _pool_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce a feature tensor to ``(batch, dim)``, keeping every row."""
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    if tensor.dim() == 2:
        return tensor
    if tensor.dim() == 3:
        # (B, seq, hidden) — CLS token
        return tensor[:, 0]
    if tensor.dim() == 4:
        _b, dim1, _dim2, dim3 = tensor.shape
        if dim1 == dim3:
            logger.warning(
                "Ambiguous 4D tensor shape %s — dim1 == dim3, assuming channels-first.",
                tensor.shape,
            )
        if dim1 >= dim3:
            return tensor.mean(dim=(-2, -1))
        return tensor.mean(dim=(1, 2))
    return tensor.reshape(tensor.shape[0], -1)


def _pool_embedding(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce an arbitrary-shape feature tensor to a 1-D embedding vector.

    Handles common output shapes:
    - (B, D)           -> take first batch item
    - (B, seq, D)      -> CLS token (index 0)
    - (B, D, H, W)     -> global average pool over spatial dims
    - (B, H, W, D)     -> channels-last, global average pool
    """
    return _pool_rows(tensor)[0]


def _recognition_vector(outputs) -> torch.Tensor:
    """Return a ``(batch, dim)`` identity embedding.

    Accepts a raw tensor of that shape, or an ``embeddings`` / ``embedding``
    attribute of that shape. ``logits``, ``pooler_output``, and
    ``last_hidden_state`` are feature maps or class scores, not a recognition
    embedding, so they are refused instead of pooled.
    """
    if isinstance(outputs, torch.Tensor):
        tensor = outputs
    else:
        tensor = None
        for name in ("embeddings", "embedding"):
            value = getattr(outputs, name, None)
            if isinstance(value, torch.Tensor):
                tensor = value
                break
        if tensor is None:
            raise ValueError(
                "embedding_head='output' requires a (batch, dim) recognition "
                "vector on the forward return (a tensor, or an .embeddings / "
                ".embedding attribute). logits, pooler_output, and "
                "last_hidden_state are not identity embeddings."
            )
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2 or tensor.shape[1] == 0:
        raise ValueError(
            "The recognition vector must have shape (batch, dim); "
            f"got {tuple(tensor.shape)}."
        )
    return tensor


class EmbeddingBackend(VisionBackend):
    """Backend that extracts penultimate-layer embeddings from any vision model.

    Returns ``{"embeddings": {"embedding": [...], "min_exemplar_distance": ...}}``.
    """

    def load(self, config):
        self._device = resolve_device(get_config_value(config, "device", "cpu"))
        model_id = get_config_value(config, "model_id")
        revision = (get_config_value(config, "revision") or "").strip() or None
        if not revision:
            raise ValueError("revision is required and must be non-empty.")

        self._model_loader = get_config_value(config, "model_loader", "transformers")
        self._hook_handle = None
        self._hooked_output = {}
        # Set before the model load so a face-identity checkpoint skips the hook.
        head = str(get_config_value(config, "embedding_head", "penultimate") or "penultimate")
        if head not in ("penultimate", "output"):
            raise ValueError("embedding_head must be 'penultimate' or 'output'.")
        self._embedding_head = head

        if self._model_loader == "transformers":
            self._load_transformers(model_id, revision)
        elif self._model_loader == "timm":
            self._load_timm(model_id, revision)
        else:
            raise ValueError(
                f"Invalid model_loader: {self._model_loader}. "
                "Must be 'transformers' or 'timm'."
            )

        # Load exemplar embeddings if provided
        self._exemplar_embeddings = None
        exemplar_path = get_config_value(config, "exemplar_embeddings_path", "")
        if exemplar_path:
            self._exemplar_embeddings = self._load_exemplars(exemplar_path)
            logger.info(
                "Loaded %d exemplar embeddings from %s",
                len(self._exemplar_embeddings),
                exemplar_path,
            )

        self._output_embeddings = get_config_value(config, "output_embeddings", True)
        self._output_distances = get_config_value(config, "output_distances", True)
        if self._embedding_head == "output":
            self._require_recognition_forward(model_id)

        logger.info(
            "EmbeddingBackend loaded model_id=%s revision=%s loader=%s device=%s",
            model_id,
            revision,
            self._model_loader,
            self._device,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_transformers(self, model_id: str, revision: str):
        """Load a HuggingFace model, preferring headless, falling back to headed + hook."""
        from transformers import AutoImageProcessor

        self._processor = AutoImageProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )

        import transformers

        self._uses_hook = False

        # Only retry on compatibility errors. OSError (disk/network) and RuntimeError
        # (e.g. OOM) must propagate — retrying them across every Auto class hides the
        # real failure and surfaces a misleading "not compatible" message.
        for cls_name in _HF_AUTO_CLASSES:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(
                    model_id, revision=revision, trust_remote_code=False, use_safetensors=True
                )
                break
            except (ValueError, TypeError, KeyError):
                continue
        else:
            raise RuntimeError(
                f"Could not load {model_id} with any supported Auto class: "
                f"{_HF_AUTO_CLASSES}"
            )

        model = model.to(self._device)
        model.eval()
        self._model = model

        # Identity checkpoints publish the recognition vector from forward.
        # A penultimate hook would keep the map before that vector.
        if getattr(self, "_embedding_head", "penultimate") == "output":
            self._uses_hook = False
            logger.info(
                "Using model forward output as the embedding (loaded via %s)",
                cls_name,
            )
            return

        # If we loaded a headed model (not plain AutoModel), install a hook
        # on the penultimate layer to capture features before the head.
        if cls_name != "AutoModel":
            self._uses_hook = True
            target = _find_penultimate_module(model)
            self._hook_handle = target.register_forward_hook(self._hook_fn)
            logger.info(
                "Installed embedding hook on %s (loaded via %s)",
                type(target).__name__,
                cls_name,
            )

    def _load_timm(self, model_id: str, revision: str | None = None):
        """Load a timm model with the classification head stripped."""
        if revision:
            logger.warning(
                "timm loader does not support 'revision' — ignoring revision=%s",
                revision,
            )
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self._model = timm.create_model(model_id, pretrained=True, num_classes=0)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._uses_hook = False

        data_config = resolve_data_config(self._model.pretrained_cfg)
        self._processor = create_transform(**data_config, is_training=False)

    def _hook_fn(self, module, input_, output):
        """Forward-hook callback: stash the module's output."""
        # output can be a tensor or a tuple/object; grab the tensor.
        if isinstance(output, torch.Tensor):
            self._hooked_output["features"] = output
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            self._hooked_output["features"] = output[0]
        elif hasattr(output, "last_hidden_state"):
            self._hooked_output["features"] = output.last_hidden_state
        else:
            self._hooked_output["features"] = output

    # ------------------------------------------------------------------
    # Exemplar loading
    # ------------------------------------------------------------------

    def _load_exemplars(self, path: str) -> np.ndarray:
        """Load exemplar embeddings from an .npz file.

        Accepts local paths and any fsspec URI (e.g. ``gs://bucket/bank.npz``).
        For ``gs://`` URIs, gcsfs auto-discovers credentials via the pod's
        workload identity / ADC. A missing or unreadable bank raises loudly
        rather than yielding a silent empty bank — an empty bank would make
        every frame look like drift.
        """
        try:
            # fsspec.open is lazy; a missing object (local or gs://) raises on
            # entering the context. Re-raise with the bank's path so a typo'd
            # URI is actionable rather than a bare fsspec path.
            with fsspec.open(path, "rb") as f:
                data = np.load(f, allow_pickle=False)
                # .npz access is lazy, so resolve the key inside the open file.
                if "embeddings" in data:
                    embeddings = data["embeddings"]
                elif "arr_0" in data:
                    embeddings = data["arr_0"]
                else:
                    keys = list(data.keys())
                    if not keys:
                        raise ValueError(f"No embeddings found in {path}")
                    embeddings = data[keys[0]]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Exemplar embeddings file not found: {path}"
            ) from e

        # A bank that loads but is empty/non-2D is the same silent-empty-bank
        # hazard as a missing file: it would crash min_exemplar_distance on
        # every frame (np.linalg.norm(..., axis=1) over a 0-row / 1D array).
        # Fail loudly here with the shape instead.
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                f"Exemplar bank at {path} must be a non-empty 2D array "
                f"(N, dim); got shape {tuple(embeddings.shape)}."
            )
        return embeddings

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _require_recognition_forward(self, model_id: str) -> None:
        """Fail at load when the checkpoint does not publish an identity vector."""
        from PIL import Image

        probe = Image.new("RGB", (112, 112), (127, 127, 127))
        try:
            self.vector(probe)
        except ValueError as exc:
            raise ValueError(
                f"{model_id} did not return a (batch, dim) recognition vector. "
                "Point embedding_model_id at a checkpoint whose forward returns "
                "that embedding. logits, pooler_output, and last_hidden_state "
                "are not identity embeddings."
            ) from exc

    def vectors(self, images) -> list[np.ndarray]:
        """Embed every image in one forward. Each row is 1-D, not yet L2-normalized."""
        images = list(images)
        if not images:
            return []
        rows = self._embed_rows(images)
        if rows.shape[0] != len(images):
            raise ValueError(
                f"Embedding forward returned {rows.shape[0]} rows for {len(images)} images."
            )
        return [np.asarray(row, dtype=np.float32).reshape(-1) for row in rows]

    def vector(self, image_pil) -> np.ndarray:
        """Return the 1-D feature vector for one image, before L2 normalization."""
        return self.vectors([image_pil])[0]

    def _embed_rows(self, images) -> np.ndarray:
        self._hooked_output.clear()
        identity = getattr(self, "_embedding_head", "penultimate") == "output"
        with torch.no_grad():
            if self._model_loader == "transformers":
                batch = images if len(images) > 1 else images[0]
                inputs = self._processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                if identity:
                    tensor = _recognition_vector(outputs)
                elif self._uses_hook and "features" in self._hooked_output:
                    tensor = _pool_rows(self._hooked_output["features"])
                elif getattr(outputs, "last_hidden_state", None) is not None:
                    tensor = _pool_rows(outputs.last_hidden_state)
                elif getattr(outputs, "pooler_output", None) is not None:
                    tensor = _pool_rows(outputs.pooler_output)
                else:
                    raise ValueError(
                        "Could not extract embedding from model output. "
                        "The model may not be supported."
                    )
            elif self._model_loader == "timm":
                stacked = torch.stack([self._processor(image) for image in images]).to(self._device)
                raw = self._model(stacked)
                tensor = _recognition_vector(raw) if identity else _pool_rows(raw)
            else:
                raise ValueError(
                    f"Invalid model_loader: {self._model_loader}. "
                    "Must be 'transformers' or 'timm'."
                )
        return tensor.detach().cpu().numpy()

    def run(self, image_pil, width, height, config):
        """Extract embedding from image.

        Returns ``{"embeddings": {"embedding": [...], ...}}``.
        """
        embedding = self.vector(image_pil)

        result = {"embeddings": {}}

        if self._output_embeddings:
            result["embeddings"]["embedding"] = embedding.tolist()

        if self._output_distances and self._exemplar_embeddings is not None:
            distances = np.linalg.norm(
                self._exemplar_embeddings - embedding, axis=1
            )
            result["embeddings"]["min_exemplar_distance"] = float(distances.min())

        return result

    def shutdown(self):
        """Release model and clear GPU memory."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._hooked_output = {}
        self._exemplar_embeddings = None
        self._processor = None
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
