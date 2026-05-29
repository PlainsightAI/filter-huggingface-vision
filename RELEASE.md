# Changelog
Huggingface Vision filter release notes


## v0.4.8 - 2026-05-29

### Added
- Configurable **class-name remapping** for object detection (PLAT-1104). The user chooses the final class name shown in `meta.detections[].class` and the visualization overlay, instead of the model's raw label (e.g. OWL-ViT's `"a handgun"`). Three ways to configure, all optional and backward compatible:
  - **Inline mapping** in `text_labels` (open-vocabulary), `finalName|||prompt` items joined by `###`: `text_labels="gun|||a handgun###gun|||a shotgun"` sends `a handgun`/`a shotgun` to the model but reports both as `gun`. Delimiters configurable via `class_delimiter` (default `|||`) / `prompt_delimiter` (default `###`). Mirrors `filter-sam3-detector`.
  - **`label_map`**: explicit `{raw: final}` rename, works for closed-vocabulary models too (e.g. `{"person": "people"}`).
  - **`collapse_labels_to`**: force every detection to a single name (e.g. `"weapon"`).
  - Precedence: `collapse_labels_to` > `label_map` > raw label. Remapping runs once per frame before meta + visualization (~6 µs / 100 detections; negligible vs inference).
  - Fail-fast config validation: empty/whitespace inline `text_labels`, malformed inline items (empty class name or prompt), and non-string `label_map` keys/values are rejected at config time. Grounding DINO can emit a sub-phrase rather than the verbatim prompt, so exact-key renames may no-op there — documented, with `collapse_labels_to` as the robust path.
- `scripts/weapon_label_remap.py`: demo pipeline showing the remap on a weapon-detection video.
- `tests/test_label_remap.py`: unit tests for parsing, validation, remap precedence, and meta/visualization agreement.

## v0.4.7 - 2026-05-20

### Fixed
- Eight silent-failure patterns in `filter-huggingface-vision` that hid broken installs, wrong results, and CUDA fallback (PLAT-889). Each path now either raises or logs at WARNING / ERROR with enough context to debug:
  - `_image_from_frame` / `_create_visualization`: removed `try/except ImportError` around PIL/numpy/cv2 (missing hard deps now fail loudly instead of silently dropping every frame / skipping visualization).
  - `OwlVitBackend` / `GroundingDinoBackend`: log a warning before returning `[]` when `text_labels` is empty or invalid at inference time.
  - `_apply_meta`: replaced `assert _dt is not None` with explicit `raise ValueError` so the check survives `python -O`.
  - `ObjectDetectionBackend` / `ImageClassificationBackend`: narrowed `except Exception` to `(ValueError, TypeError, KeyError)` so `OSError` / `MemoryError` / `ConnectionError` propagate instead of being relabeled as "model not compatible".
  - `_image_from_frame` cv2.cvtColor: catch only `cv2.error` (with log) and propagate the rest.
  - `process()`: log a warning when frames pass through with no initialized backend.

### Added
- `tests/test_silent_failures.py`: 17 unit tests covering every new raise/log path.

## v0.4.6 - 2026-05-01

### Fixed
- `open-vocabulary` backend now supports OWLv2 models (`google/owlv2-*`): replaced hardcoded `OwlViTProcessor` + `OwlViTForObjectDetection` with `AutoProcessor` + `AutoModelForZeroShotObjectDetection`. Loading an OWLv2 checkpoint with the OWLv1 class silently initialized all weights from random, producing zero-quality detections with no error raised.

### Changed
- `open-vocabulary` backend loads models in `float16` by default (~3× throughput improvement on GPU vs fp32 baseline).
- `open-vocabulary` backend logs a single FPS summary line on `shutdown()` for throughput observability.
- `scripts/zero_shot_object_detection.py` updated to use `google/owlv2-base-patch16-ensemble` and `!sync!resize=960x540` VideoIn option.
- `docker-compose.yaml`: add `env_file`, `FILTER_DETECTION_TYPE`, `FILTER_DEVICE`, `FILTER_MAX_DETECTIONS`, `FILTER_TOP_K`, `FILTER_TEXT_LABELS`; VideoIn source now uses `resize=960x540!sync` and supports `VIDEO_INPUT` / `VIDEO_PATH` overrides.

## v0.4.5 - 2026-04-23

### Changed
- Bump openfilter SDK, align CI workflow with shared release gate (source-paths)

- Fix release workflow secret names: `PYPI_API_TOKEN` → `PLAINSIGHT_PYPI_TOKEN`, `DOCKERHUB_TOKEN` → `DOCKERHUB_ACCESS_TOKEN` (org-level secret names). Without this the PyPI / Docker Hub tokens resolved to empty and no package has been published since the migration.
- Bump openfilter dependency to `>=0.1.30`.

## [Unreleased]

### Changed

- Bump openfilter to 1.1.0

## v0.4.4 - 2026-04-20

### Changed
- Remove redundant ci.yaml (shared workflow handles PR testing)
- Add push + pull_request triggers to create-release.yaml


## v0.4.3 - 2026-04-20

### Changed
- Replace inline create-release.yaml with shared workflow caller (~13 lines)
- Switch to shared security-scan workflow
- Bump openfilter to >=0.1.27
- Secret names updated to PYPI_API_TOKEN / DOCKERHUB_TOKEN


## v0.4.2 - 2026-04-10

### Fixed
- Docker image: pre-install a pinned **torch + torchvision** stack (`2.9.1+cu128` on `linux/amd64`) and apply **pip constraints** on the final install so transitive dependencies cannot upgrade PyTorch to a CUDA 13 line incompatible with current production GPUs (driver cap at CUDA 12.8). `linux/arm64` builds use the PyPI CPU/default stack (e.g. `2.9.1+cpu`).

### Changed
- Require `torch>=2.9.1` in `pyproject.toml` (PyPI-safe; the CUDA 12.8 wheel remains Docker-only).

## v0.4.1 - 2026-04-03

### Fixed
- Rebuild with openfilter v0.1.27 which removes eager imports from `filters/__init__.py` that crashed containers without optional dependencies (e.g. PyAV).

## v0.4.0 - 2026-04-01

### Added
- Embedding extraction backend (`detection_type="embedding"`) — model-agnostic penultimate-layer feature extraction using PyTorch forward hooks. Works with any HuggingFace Transformers vision model (`AutoModel`, `AutoModelForImageClassification`, `AutoModelForObjectDetection`, etc.) and timm models via `model_loader` config.
- Exemplar distance computation: optional L2 distance to closest reference embedding for similarity-based anomaly detection (`exemplar_embeddings_path`, `output_distances`).
- New config options: `model_loader`, `exemplar_embeddings_path`, `output_embeddings`, `output_distances`.
- `scripts/generate_exemplars.py` — offline script to extract exemplar embeddings from reference images into `.npz`.
- Embedding section in `docs/supported-models.md` with example model IDs, config, and output format.
- Comprehensive test suite for embedding backend (`tests/test_embedding.py`).

### Changed
- Updated README, `docs/overview.md`, and `docs/supported-models.md` with embedding extraction documentation.
- Backend base class (`VisionBackend.run()`) documents third return type for embeddings.
- Filter `process()` writes embedding results to `frame.data` directly (not nested under `meta`).
- Simplified `pyproject.toml` uv index config (marked openfilter index as explicit).

### Fixed
- `draw_visualization` warning emitted once at setup instead of per frame when used with unsupported detection types.

## v0.3.2 - 2026-03-14
### Fixed
- Fixed `ModuleNotFoundError: No module named 'filter_huggingface_vision.backends'` caused by `pyproject.toml` excluding sub-packages from the built wheel. Switched from explicit package list to setuptools auto-discovery.

## v0.3.1 - 2026-03-13
### Changed
- Added pre-build disk cleanup in `publish-docker` GitHub Actions job to reduce runner disk usage before multi-arch builds
- Tuned Docker Buildx cache export from `mode=max` to `mode=min` to lower temporary disk pressure during release builds

## v0.3.0 - 2026-03-12
### Changed
- Open-sourced filter under Apache 2.0 license
- Replaced private CI/CD with public GitHub Actions workflows (CI, release automation, security scanning)
- Replaced private base image Dockerfile with multi-stage public build from PyPI
- Replaced private Google Artifact Registry references with public PyPI and Docker Hub
- Updated Makefile to self-contained targets (removed build-include dependency)
- Updated pyproject.toml with public packaging configuration (Python 3.10-3.13, Apache-2.0 license)
- Updated docker-compose.yaml to use public Docker Hub images
- Updated dependencies (openfilter>=0.1.21, black, isort, flake8)

### Added
- Apache 2.0 LICENSE file
- Developer Certificate of Origin (DCO)
- CONTRIBUTING.md with development setup, PR process, and release workflow
- MAINTAINERS.md with core maintainer list
- CODEOWNERS for code review routing
- Pull request template
- MANIFEST.in for package distribution
- .env.example with configuration reference
- .flake8 linting configuration
- Release automation workflow (PyPI + Docker Hub publishing)
- Security scanning workflow (Grype, weekly schedule)

### Removed
- Private build-include/ shared Makefile dependency
- Private Google Artifact Registry (GAR) configuration
- Private base image references (filter_base)
- Dockerfile.model (private model image)
- RESOURCE_BUNDLE_VERSION (private versioning)
- models.toml, models.toml.example, prepare_models.py (private model bundling)

## v0.2.0 - 2026-02-13
### Added
- Image classification backend (AutoImageProcessor + AutoModelForImageClassification); supports ViT and ConvNeXt (e.g. `google/vit-base-patch16-224`, `facebook/convnext-tiny-224`)
- New `detection_type="image-classification"` with config `top_k`; output in `frame.data["meta"]` with `classification` (architecture, classes, confidences)
- Optional visualization for classification (top label + score as text on image)
- Script `scripts/image_classification.py` (VideoIn → FilterHuggingfaceVision → Webvis)

## v0.1.0 - 2026-02-10

### Added
- Initial release: Hugging Face Vision filter for OpenFilter
- Object detection via Hugging Face Transformers (AutoImageProcessor + AutoModelForObjectDetection)
- Configurable model_id, revision, threshold, device, and max_detections
- Output in `frame.data["meta"]` with `detections` (list of `{class, rois}` normalized), `detection_confidence`
- Optional visualization topic with bounding boxes and labels drawn on the image
- Frame input via OpenFilter convention (`frame.rw_bgr.image`) with fallback to `frame.data[topic]`
- Object detection pipeline script (VideoIn → FilterHuggingfaceVision → Webvis)
- Support for RT-DETR and DETR-style models (dict and object detection outputs)
