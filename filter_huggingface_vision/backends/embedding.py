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
import os

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


def _pool_embedding(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce an arbitrary-shape feature tensor to a 1-D embedding vector.

    Handles common output shapes:
    - (B, D)           -> take first batch item
    - (B, seq, D)      -> CLS token (index 0)
    - (B, D, H, W)     -> global average pool over spatial dims
    - (B, H, W, D)     -> channels-last, global average pool
    """
    if tensor.dim() == 2:
        return tensor[0]
    if tensor.dim() == 3:
        # (B, seq_len, hidden) — take CLS token
        return tensor[0, 0]
    if tensor.dim() == 4:
        b, dim1, dim2, dim3 = tensor.shape
        # Heuristic: channels-first if dim1 > dim3 (e.g. (B,768,14,14) has
        # 768 > 14 so channels-first; (B,14,14,768) has 14 < 768 so channels-last).
        if dim1 == dim3:
            logger.warning(
                "Ambiguous 4D tensor shape %s — dim1 == dim3, assuming channels-first.",
                tensor.shape,
            )
        if dim1 >= dim3:
            # channels-first: (B, D, H, W)
            return tensor[0].mean(dim=(-2, -1))
        else:
            # channels-last: (B, H, W, D)
            return tensor[0].mean(dim=(0, 1))
    # Fallback: flatten and hope for the best
    return tensor[0].flatten()


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
                    model_id, revision=revision, trust_remote_code=False
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
        """Load exemplar embeddings from an .npz file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Exemplar embeddings file not found: {path}")

        data = np.load(path)
        if "embeddings" in data:
            return data["embeddings"]
        elif "arr_0" in data:
            return data["arr_0"]
        else:
            keys = list(data.keys())
            if keys:
                return data[keys[0]]
            raise ValueError(f"No embeddings found in {path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, image_pil, width, height, config):
        """Extract embedding from image.

        Returns ``{"embeddings": {"embedding": [...], ...}}``.
        """
        self._hooked_output.clear()

        with torch.no_grad():
            if self._model_loader == "transformers":
                inputs = self._processor(images=image_pil, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._model(**inputs)

                if self._uses_hook and "features" in self._hooked_output:
                    raw = self._hooked_output["features"]
                elif hasattr(outputs, "last_hidden_state"):
                    raw = outputs.last_hidden_state
                elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    raw = outputs.pooler_output
                else:
                    raise ValueError(
                        "Could not extract embedding from model output. "
                        "The model may not be supported."
                    )

                embedding = _pool_embedding(raw).cpu().numpy()

            elif self._model_loader == "timm":
                tensor = self._processor(image_pil).unsqueeze(0).to(self._device)
                raw = self._model(tensor)
                embedding = _pool_embedding(raw).cpu().numpy()

        embedding = embedding.squeeze()

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
