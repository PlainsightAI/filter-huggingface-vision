#!/usr/bin/env python

"""
Zero-Shot Object Detection Pipeline (OWL-ViT / OWLv2).

Pipeline: VideoIn → FilterHuggingfaceVision (zero-shot-object-detection) → Webvis

Uses model google/owlv2-base-patch16-ensemble by default (fixed; do not set MODEL_ID in .env for this script).
Required env: VIDEO_PATH
Optional: THRESHOLD (default 0.1), PORT (default 8010)
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

# Default model: OWLv2 (better accuracy than OWLv1; both work with open-vocabulary detection_type)
# MODEL_ID = "google/owlvit-base-patch32"
# MODEL_ID = "google/owlvit-base-patch16"
MODEL_ID = "google/owlv2-base-patch16-ensemble"
REVISION = "main"
# Text queries for zero-shot detection (one list per image)
TEXT_LABELS = [["person", "cup"]]

if __name__ == "__main__":
    video_path = os.getenv("VIDEO_PATH", "")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError("VIDEO_PATH must point to an existing video. Set it in .env.")
    threshold = float(os.getenv("THRESHOLD", "0.1"))
    port = int(os.getenv("PORT", "8010"))

    print("Pipeline: Zero-Shot (OWL-ViT / OWLv2)")
    print(f"Video: {video_path} | Model: {MODEL_ID} @ {REVISION} | Labels: {TEXT_LABELS}")

    Filter.run_multi(
        [
            (VideoIn, dict(sources=f"file://{video_path}!sync!resize=960x540", outputs="tcp://*:5550")),
            (
                FilterHuggingfaceVision,
                FilterHuggingfaceVisionConfig(
                    id="filter_huggingface_vision",
                    sources="tcp://localhost:5550",
                    outputs="tcp://*:5552",
                    model_id=MODEL_ID,
                    revision=REVISION,
                    detection_type="open-vocabulary",
                    text_labels=TEXT_LABELS,
                    threshold=threshold,
                    top_k=15,
                    draw_visualization=True,
                    device="cuda",
                    visualization_topic="viz",
                ),
            ),
            (
                Webvis,
                dict(id="webvis", sources="tcp://localhost:5552;main,tcp://localhost:5552;viz", port=port),
            ),
        ]
    )
