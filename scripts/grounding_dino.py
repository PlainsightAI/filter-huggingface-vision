#!/usr/bin/env python

"""
Grounding DINO Pipeline (open-vocabulary).

Pipeline: VideoIn → FilterHuggingfaceVision (open-vocabulary-grounding) → Webvis

Uses a fixed Grounding DINO / MM Grounding DINO model (do not set MODEL_ID in .env for this script).
Required env: VIDEO_PATH
Optional: THRESHOLD (default 0.3), PORT (default 8010)

Example .env:
    VIDEO_PATH=./filter_example_video.mp4
    THRESHOLD=0.3
    PORT=8010
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

# Grounding DINO model (fixed in script to avoid loading wrong model)
# Same as HF example: IDEA-Research/grounding-dino-tiny (boxes scale correctly)
# Alternative: openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
REVISION = "main"
# Text queries for open-vocabulary detection (one list per image)
TEXT_LABELS = [["a person", "a cup", "a cat"]]


if __name__ == "__main__":
    video_path = os.getenv("VIDEO_PATH", "")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError("VIDEO_PATH must point to an existing video. Set it in .env.")
    threshold = float(os.getenv("THRESHOLD", "0.3"))
    port = int(os.getenv("PORT", "8010"))

    print("Pipeline: Grounding DINO (open-vocabulary-grounding)")
    print(f"Video: {video_path} | Model: {MODEL_ID} @ {REVISION} | Labels: {TEXT_LABELS}")

    Filter.run_multi(
        [
            (VideoIn, dict(sources=f"file://{video_path}!loop", outputs="tcp://*:5550")),
            (
                FilterHuggingfaceVision,
                FilterHuggingfaceVisionConfig(
                    id="filter_huggingface_vision",
                    sources="tcp://localhost:5550",
                    outputs="tcp://*:5552",
                    model_id=MODEL_ID,
                    revision=REVISION,
                    detection_type="open-vocabulary-grounding",
                    text_labels=TEXT_LABELS,
                    threshold=threshold,
                    draw_visualization=True,
                    visualization_topic="viz",
                ),
            ),
            (
                Webvis,
                dict(id="webvis", sources="tcp://localhost:5552;main,tcp://localhost:5552;viz", port=port),
            ),
        ]
    )
