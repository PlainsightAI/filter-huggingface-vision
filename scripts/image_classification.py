#!/usr/bin/env python

"""
Image Classification Pipeline using FilterHuggingfaceVision.

This script demonstrates a simple pipeline that:
1. Reads input video
2. Runs Hugging Face image classification on video frames (AutoImageProcessor + AutoModelForImageClassification)
3. Provides web-based visualization (top label + score)

Pipeline: VideoIn → FilterHuggingfaceVision (Image Classification) → Webvis

Required environment variables (e.g. in .env):
    MODEL_ID: Hugging Face model id (e.g. google/vit-base-patch16-224 or facebook/convnext-tiny-224)
    REVISION: Model revision (required for reproducibility)
    VIDEO_PATH: Path to the input video file

Optional environment variables:
    TOP_K: Number of top classes to return (default: 5)
    PORT: Webvis port (default: 8010)
    DRAW_VISUALIZATION: If "true", add a "viz" topic with top label drawn (default: false)

Example .env:
    MODEL_ID=google/vit-base-patch16-224
    REVISION=main
    VIDEO_PATH=./filter_example_video.mp4
    TOP_K=5
    PORT=8010

Output:
    frame.data["meta"] with detections, detection_confidence, classification (architecture, classes, confidences). Upstream meta preserved.
"""

import os

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    load_dotenv()
    # Load .env from project root (parent of scripts/) so it works regardless of cwd
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    load_dotenv(os.path.join(_project_root, ".env"))
except ImportError:
    pass

from openfilter.filter_runtime.filter import Filter
from openfilter.filter_runtime.filters.video_in import VideoIn
from openfilter.filter_runtime.filters.webvis import Webvis

from filter_huggingface_vision.filter import (
    FilterHuggingfaceVision,
    FilterHuggingfaceVisionConfig,
)


if __name__ == "__main__":
    video_path = os.getenv("VIDEO_PATH", "")
    if not video_path:
        raise ValueError(
            "VIDEO_PATH environment variable is not set. Set it in your .env or environment."
        )
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}\nSet VIDEO_PATH correctly in .env or environment."
        )

    model_id = os.getenv("MODEL_ID", "")
    revision = os.getenv("REVISION", "")
    if not model_id:
        raise ValueError(
            "MODEL_ID environment variable is not set (e.g. google/vit-base-patch16-224)."
        )
    if not revision:
        raise ValueError(
            "REVISION environment variable is not set (required for reproducibility)."
        )

    top_k = int(os.getenv("TOP_K", "5"))
    port = int(os.getenv("PORT", "8010"))

    print("Running Image Classification Pipeline (Hugging Face Vision)")
    print(f"Video source: {video_path}")
    print(f"Model: {model_id} @ {revision}")
    print("Pipeline: VideoIn → FilterHuggingfaceVision (Image Classification) → Webvis")

    Filter.run_multi(
        [
            (
                VideoIn,
                dict(
                    sources=f"file://{video_path}!loop",
                    outputs="tcp://*:5550",
                ),
            ),
            (
                FilterHuggingfaceVision,
                FilterHuggingfaceVisionConfig(
                    id="filter_huggingface_vision",
                    sources="tcp://localhost:5550",
                    outputs="tcp://*:5552",
                    model_id=model_id,
                    revision=revision,
                    detection_type="image-classification",
                    top_k=top_k,
                    draw_visualization=True,
                    visualization_topic="viz",
                ),
            ),
            (
                Webvis,
                dict(
                    id="webvis",
                    sources="tcp://localhost:5552;main,tcp://localhost:5552;viz",
                    port=port,
                ),
            ),
        ]
    )
