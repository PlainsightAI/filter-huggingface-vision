# syntax=docker/dockerfile:1.4
FROM python:3.13-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ARG TARGETPLATFORM
# Bump these together when upgrading PyTorch; used for install, constraints, and CUDA wheel index path.
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG CUDA_SUFFIX=cu128
# Optional: pin the wheel version for local smoke tests before that release exists on PyPI (e.g. --build-arg FILTER_PKG_VERSION_OVERRIDE=0.4.1).
ARG FILTER_PKG_VERSION_OVERRIDE=

RUN --mount=type=bind,source=VERSION,target=/tmp/VERSION,ro \
    set -eux; \
    RAW="$(head -n1 /tmp/VERSION)"; \
    PKG_VERSION="$(printf '%s' "$RAW" | tr -d ' \t\r\n' | sed 's/^[vV]//')"; \
    [ -n "$PKG_VERSION" ] || { echo "VERSION file is empty"; exit 1; }; \
    INSTALL_VER="${FILTER_PKG_VERSION_OVERRIDE:-$PKG_VERSION}"; \
    INSTALL_VER="$(printf '%s' "$INSTALL_VER" | tr -d ' \t\r\n' | sed 's/^[vV]//')"; \
    [ -n "$INSTALL_VER" ] || { echo "filter package version is empty after normalization; check VERSION and FILTER_PKG_VERSION_OVERRIDE"; exit 1; }; \
    pip install --no-cache-dir --upgrade pip && \
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
      echo "Installing CUDA ${CUDA_SUFFIX} PyTorch (${TORCH_VERSION}+${CUDA_SUFFIX}, torchvision ${TORCHVISION_VERSION})..."; \
      pip install --no-cache-dir "torch==${TORCH_VERSION}+${CUDA_SUFFIX}" "torchvision==${TORCHVISION_VERSION}" \
        --extra-index-url "https://download.pytorch.org/whl/${CUDA_SUFFIX}"; \
      printf '%s\n' "torch==${TORCH_VERSION}+${CUDA_SUFFIX}" "torchvision==${TORCHVISION_VERSION}" > /tmp/pip-constraints.txt; \
      PYTORCH_EXTRA="--extra-index-url https://download.pytorch.org/whl/${CUDA_SUFFIX}"; \
    else \
      echo "WARNING: Non-amd64 or unset TARGETPLATFORM='${TARGETPLATFORM}' — installing CPU-only torch"; \
      pip install --no-cache-dir "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
      printf '%s\n' "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" > /tmp/pip-constraints.txt; \
      PYTORCH_EXTRA=""; \
    fi && \
    pip install --no-cache-dir \
    -c /tmp/pip-constraints.txt \
    --index-url https://python.openfilter.io/simple \
    $PYTORCH_EXTRA \
    --extra-index-url https://pypi.org/simple \
    "filter-huggingface-vision==${INSTALL_VER}"

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
