# Changelog
Huggingface Vision filter release notes

## [Unreleased]

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
