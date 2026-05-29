#!/usr/bin/env python

"""
Class-name remap demo (open-vocabulary / OWL-ViT).

Pipeline: VideoIn → FilterHuggingfaceVision (open-vocabulary) → Webvis

Shows the PLAT-1104 feature: detection models return verbose prompt labels
("a handgun", "a shotgun"); this filter lets the user pick the final name shown
in meta.detections[].class and the visualization overlay.

Two ways to rename, configured here via env:
  - Inline mapping in TEXT_LABELS using "finalName|||prompt" items joined by "###":
        TEXT_LABELS="gun|||a handgun###gun|||a shotgun"
    The model still receives "a handgun"/"a shotgun"; the output class becomes "gun".
  - Collapse everything into one name:
        COLLAPSE_LABELS_TO="weapon"

Required env: VIDEO_PATH
Optional env: THRESHOLD (default 0.1), PORT (default 8010),
              TEXT_LABELS (default below), COLLAPSE_LABELS_TO (default unset)
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

MODEL_ID = "google/owlvit-base-patch32"
REVISION = "main"

if __name__ == "__main__":
    video_path = os.getenv("VIDEO_PATH", "")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError("VIDEO_PATH must point to an existing video. Set it in .env.")

    threshold = float(os.getenv("THRESHOLD", "0.1"))
    port = int(os.getenv("PORT", "8010"))
    # Default: detect several weapon prompts but report them all as "gun".
    text_labels = os.getenv(
        "TEXT_LABELS",
        "gun|||a handgun###gun|||a shotgun###gun|||a pistol###gun|||a rifle",
    )
    collapse_labels_to = os.getenv("COLLAPSE_LABELS_TO") or None

    print("Pipeline: class-name remap demo (OWL-ViT, open-vocabulary)")
    print(f"Video: {video_path} | Model: {MODEL_ID} @ {REVISION}")
    print(f"TEXT_LABELS={text_labels!r} | COLLAPSE_LABELS_TO={collapse_labels_to!r}")

    Filter.run_multi(
        [
            (VideoIn, dict(sources=f"file://{video_path}!loop!sync", outputs="tcp://*:5550")),
            (
                FilterHuggingfaceVision,
                FilterHuggingfaceVisionConfig(
                    id="filter_huggingface_vision",
                    sources="tcp://localhost:5550",
                    outputs="tcp://*:5552",
                    model_id=MODEL_ID,
                    revision=REVISION,
                    detection_type="open-vocabulary",
                    text_labels=text_labels,
                    collapse_labels_to=collapse_labels_to,
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
