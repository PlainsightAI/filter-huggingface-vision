# syntax=docker/dockerfile:1.4
FROM python:3.13-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Prefer buildx so TARGETPLATFORM is set (e.g. docker buildx build --platform linux/amd64). Plain docker build
# may leave TARGETPLATFORM empty; we fall back to BUILDPLATFORM for branch selection only.
ARG TARGETPLATFORM
ARG BUILDPLATFORM
# Bump these together when upgrading PyTorch; used for install, constraints, and CUDA wheel index path.
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG CUDA_SUFFIX=cu128
# Optional: pin the wheel version for local smoke tests before that release exists on PyPI (e.g. --build-arg FILTER_PKG_VERSION_OVERRIDE=0.4.1).
ARG FILTER_PKG_VERSION_OVERRIDE=

RUN --mount=type=bind,source=VERSION,target=/tmp/VERSION,ro \
    --mount=type=bind,target=/src,ro \
    set -eux; \
    TP="${TARGETPLATFORM}"; BP="${BUILDPLATFORM}"; \
    if [ -z "$TP" ]; then PLAT="$BP"; else PLAT="$TP"; fi; \
    RAW="$(head -n1 /tmp/VERSION)"; \
    PKG_VERSION="$(printf '%s' "$RAW" | tr -d ' \t\r\n' | sed 's/^[vV]//')"; \
    [ -n "$PKG_VERSION" ] || { echo "VERSION file is empty"; exit 1; }; \
    INSTALL_VER="${FILTER_PKG_VERSION_OVERRIDE:-$PKG_VERSION}"; \
    INSTALL_VER="$(printf '%s' "$INSTALL_VER" | tr -d ' \t\r\n' | sed 's/^[vV]//')"; \
    [ -n "$INSTALL_VER" ] || { echo "filter package version is empty after normalization; check VERSION and FILTER_PKG_VERSION_OVERRIDE"; exit 1; }; \
    pip install --no-cache-dir --upgrade pip && \
    if [ "$PLAT" = "linux/amd64" ]; then \
      echo "Installing CUDA ${CUDA_SUFFIX} PyTorch (${TORCH_VERSION}+${CUDA_SUFFIX}, torchvision ${TORCHVISION_VERSION}+${CUDA_SUFFIX})..."; \
      pip install --no-cache-dir "torch==${TORCH_VERSION}+${CUDA_SUFFIX}" "torchvision==${TORCHVISION_VERSION}+${CUDA_SUFFIX}" \
        --extra-index-url "https://download.pytorch.org/whl/${CUDA_SUFFIX}"; \
      printf '%s\n' "torch==${TORCH_VERSION}+${CUDA_SUFFIX}" "torchvision==${TORCHVISION_VERSION}+${CUDA_SUFFIX}" > /tmp/pip-constraints.txt; \
      PYTORCH_EXTRA="--extra-index-url https://download.pytorch.org/whl/${CUDA_SUFFIX}"; \
    else \
      echo "WARNING: effective platform is not linux/amd64 (PLAT=$PLAT, TARGETPLATFORM was $TP, BUILDPLATFORM was $BP) — installing CPU-only torch"; \
      pip install --no-cache-dir "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
      printf '%s\n' "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" > /tmp/pip-constraints.txt; \
      PYTORCH_EXTRA=""; \
    fi && \
    # Install the filter: prefer the published wheel (release / post-publish builds), but fall back to
    # the local source tree if that version isn't on the index yet (e.g. dry-run-publish on a PR that
    # bumps VERSION ahead of PyPI). Both paths use the same pip constraints, so torch/torchvision pins
    # are honored regardless of source.
    if pip install --no-cache-dir \
        -c /tmp/pip-constraints.txt \
        --index-url https://python.openfilter.io/simple \
        $PYTORCH_EXTRA \
        --extra-index-url https://pypi.org/simple \
        "filter-huggingface-vision[gcs]==${INSTALL_VER}"; then \
      echo "Installed filter-huggingface-vision[gcs]==${INSTALL_VER} from index"; \
    else \
      echo "filter-huggingface-vision[gcs]==${INSTALL_VER} not on index yet; falling back to local source (dry-run / pre-publish build)"; \
      # /src is bind-mounted read-only; setuptools writes egg-info during the build, so copy first. \
      cp -r /src /tmp/build && \
      pip install --no-cache-dir \
        -c /tmp/pip-constraints.txt \
        --index-url https://python.openfilter.io/simple \
        $PYTORCH_EXTRA \
        --extra-index-url https://pypi.org/simple \
        "/tmp/build[gcs]"; \
    fi && \
    if [ "$PLAT" = "linux/amd64" ]; then \
      python -c "import torch; assert torch.version.cuda is not None, 'CUDA torch not installed'"; \
    fi

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libxcb-shm0 libxcb-render0 libx11-6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash appuser
WORKDIR /app

RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

COPY --from=builder /usr/local /usr/local

CMD ["python", "-m", "filter_huggingface_vision.filter"]
