#!/usr/bin/env python

"""
Generate Exemplar Embeddings for the Embedding Backend.

This script extracts embeddings from a directory of reference images and saves
them as a .npz file that can be loaded by the embedding backend via the
`exemplar_embeddings_path` config option.

The resulting exemplar embeddings are used at runtime to compute L2 distances
between live frame embeddings and the reference set — the minimum distance
indicates how "similar" a frame is to the closest exemplar.

Usage:
    python scripts/generate_exemplars.py

Required environment variables (or in .env):
    MODEL_ID:       HuggingFace model id or timm model name
    REVISION:       Model revision (required for HF models; ignored for timm)
    IMAGE_DIR:      Directory containing reference images (.jpg, .jpeg, .png)

Optional environment variables:
    MODEL_LOADER:   "transformers" (default) or "timm"
    DEVICE:         "cpu" (default) or "cuda"
    OUTPUT_PATH:    Output .npz file (default: exemplars.npz)

Example .env (HuggingFace model):
    MODEL_ID=facebook/dinov2-small
    REVISION=main
    IMAGE_DIR=./reference_images
    OUTPUT_PATH=./exemplars.npz

Example .env (timm model):
    MODEL_ID=convnext_tiny.dinov3_lvd1689m
    REVISION=main
    IMAGE_DIR=./reference_images
    MODEL_LOADER=timm
    OUTPUT_PATH=./exemplars.npz
"""

import glob
import logging
import os
import sys

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Reuse the backend's hook-based extraction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from filter_huggingface_vision.backends.embedding import EmbeddingBackend, _pool_embedding


def _collect_images(image_dir: str) -> list[str]:
    """Collect image paths from a directory."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
        paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    paths = sorted(set(paths))
    return paths


def main():
    model_id = os.environ.get("MODEL_ID")
    revision = os.environ.get("REVISION", "main")
    image_dir = os.environ.get("IMAGE_DIR")
    model_loader = os.environ.get("MODEL_LOADER", "transformers")
    device = os.environ.get("DEVICE", "cpu")
    output_path = os.environ.get("OUTPUT_PATH", "exemplars.npz")

    if not model_id:
        logger.error("MODEL_ID is required. Set it in .env or as an environment variable.")
        sys.exit(1)
    if not image_dir:
        logger.error("IMAGE_DIR is required. Set it in .env or as an environment variable.")
        sys.exit(1)
    if not os.path.isdir(image_dir):
        logger.error(f"IMAGE_DIR does not exist: {image_dir}")
        sys.exit(1)

    image_paths = _collect_images(image_dir)
    if not image_paths:
        logger.error(f"No images found in {image_dir}")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} images in {image_dir}")
    logger.info(f"Model: {model_id} (loader={model_loader}, device={device})")

    # Build a minimal config dict for the backend
    config = {
        "model_id": model_id,
        "revision": revision,
        "model_loader": model_loader,
        "device": device,
        "exemplar_embeddings_path": "",
        "output_embeddings": True,
        "output_distances": False,
    }

    # Load the backend (reuses the same hook-based extraction as runtime)
    backend = EmbeddingBackend()
    backend.load(type("Config", (), {k: v for k, v in config.items()})())

    embeddings = []
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            result = backend.run(img, w, h, config)
            embedding = result["embeddings"]["embedding"]
            embeddings.append(embedding)
            logger.info(f"  [{i+1}/{len(image_paths)}] {os.path.basename(path)} -> dim={len(embedding)}")
        except Exception as e:
            logger.warning(f"  [{i+1}/{len(image_paths)}] {os.path.basename(path)} FAILED: {e}")

    backend.shutdown()

    if not embeddings:
        logger.error("No embeddings extracted. Check your images and model.")
        sys.exit(1)

    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.savez(output_path, embeddings=embeddings_array)
    logger.info(
        f"Saved {len(embeddings)} exemplar embeddings "
        f"(shape={embeddings_array.shape}) to {output_path}"
    )


if __name__ == "__main__":
    main()
