#!/usr/bin/env python

"""
Object Detection Pipeline using FilterHuggingfaceVision.

This script demonstrates a simple pipeline that:
1. Reads input video
2. Runs Hugging Face object detection on video frames (AutoImageProcessor + AutoModelForObjectDetection)
3. Provides web-based visualization of detections

Pipeline: VideoIn → FilterHuggingfaceVision (Object Detection) → Webvis

Required environment variables (e.g. in .env):
    MODEL_ID: Hugging Face model id (e.g. PekingU/rtdetr_r50vd)
    REVISION: Model revision (required for reproducibility)
    VIDEO_PATH: Path to the input video file

Optional environment variables:
    THRESHOLD: Detection confidence threshold in [0,1] (default: 0.3)
    PORT: Webvis port (default: 8010)
    DRAW_VISUALIZATION: If "true", add a "viz" topic with bounding boxes drawn (default: false)

Example .env:
    MODEL_ID=PekingU/rtdetr_r50vd
    REVISION=main
    VIDEO_PATH=./filter_example_video.mp4
    THRESHOLD=0.3
    PORT=8010

Output:
    frame.data["subjects"]["huggingface_vision"] with detection_type, model, image, detections (label, score, box xyxy).
"""

import os

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    load_dotenv()
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
            "MODEL_ID environment variable is not set (e.g. PekingU/rtdetr_r50vd)."
        )
    if not revision:
        raise ValueError(
            "REVISION environment variable is not set (required for reproducibility)."
        )

    threshold = float(os.getenv("THRESHOLD", "0.3"))
    port = int(os.getenv("PORT", "8010"))

    print("Running Object Detection Pipeline (Hugging Face Vision)")
    print(f"Video source: {video_path}")
    print(f"Model: {model_id} @ {revision}")
    print("Pipeline: VideoIn → FilterHuggingfaceVision (Object Detection) → Webvis")

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
                    detection_type="closed-vocabulary",
                    threshold=threshold,
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
