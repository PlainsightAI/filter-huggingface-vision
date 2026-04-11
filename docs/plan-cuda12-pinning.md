# Plan: Pin filter-huggingface-vision to CUDA 12.x PyTorch wheels

## Implementation status

**Shipped in v0.4.2 (2026-04-10).** The builder stage in `Dockerfile` declares `ARG TARGETPLATFORM` and, before `pip install filter-huggingface-vision`, installs `torch==2.9.1+cu128` from `https://download.pytorch.org/whl/cu128` on `linux/amd64` and `torch==2.9.1` on other platforms (e.g. `linux/arm64`). `pyproject.toml` sets `torch>=2.9.1` without a CUDA suffix so PyPI installs stay flexible.

## Context

`filter-huggingface-vision` installs `torch` without any version or CUDA constraint. During a Docker image build, pip resolves the latest available wheel — which is now built for CUDA 13.0. All current GPU targets (Beast A10 with driver 570.x, GKE T4 with driver 580.x) cap at CUDA 12.8, so the image silently becomes incompatible with production.

The fix must live in the Dockerfile, **not** in `pyproject.toml`'s package dependencies, because pinning a CUDA-specific wheel (`torch+cu128`) in the published PyPI package would break every user installing on CPU or with a different CUDA version.

---

## Complexity breakdown

### 1. PyTorch is an unpinned transitive dependency
`pyproject.toml` declares `torch` (no version, no CUDA suffix). pip resolves it at image-build time by pulling the highest-versioned wheel available — currently CUDA 13.0. This is the direct cause of the incident.

### 2. Multi-platform builds require conditional wheel selection
The `create-release.yaml` workflow builds `linux/amd64` AND `linux/arm64` (Makefile `build-image` target). CUDA wheels (`torch+cu128`) only exist for `amd64`. The `arm64` platform (Mac M-series CI, etc.) needs the CPU/default wheel. Without a platform conditional, the `arm64` build would fail trying to install a non-existent CUDA wheel.

### 3. PyPI package compatibility must be preserved
The package is published to PyPI and installed by external users. Pinning `torch==X.Y.Z+cu128` in `pyproject.toml` would force every pip install to target CUDA 12.8 — breaking CPU users and other CUDA versions. The CUDA pin must be Docker-only.

### 4. Base image approach is architecturally messier here
Using `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` as base image (the reference approach from filter-sam3-detector) works well for that repo's simple structure, but introduces friction here:
- The PyTorch Docker images are Ubuntu-based, while the current Dockerfile uses Debian slim — different system library set, different apt package names.
- They ship a fixed Python version (3.11 as of 2.9.x), conflicting with the project's stated Python 3.13 support and the current `python:3.13-slim` base.
- They are multi-GB images vs. the current slim setup.

The wheel-pinning approach (pre-install torch before the package) achieves the same guarantee without restructuring the image.

### 5. uv.lock is not used at Docker build time
The Dockerfile installs via pip from the custom OpenFilter index. `uv.lock` governs the development workflow (`make install`), not the Docker build. No lock-file changes are needed for the Dockerfile fix.

---

## Recommended approach: Explicit wheel pre-install in the Dockerfile builder stage

### Files to modify

| File | Change |
|------|--------|
| `Dockerfile` | Add `TARGETPLATFORM` conditional torch pre-install in builder stage |
| `pyproject.toml` | Add minimum version floor `torch>=2.9.1` (no CUDA suffix) |

### Dockerfile change — builder stage

Add a `TARGETPLATFORM` build arg (Docker sets this automatically during `buildx` multi-platform builds) and conditionally install the right torch wheel before installing the package. pip will then see torch as already satisfied and skip reinstalling it.

```dockerfile
ARG TARGETPLATFORM

# Install PyTorch pinned to CUDA 12.8 on amd64; fall back to default wheel on arm64
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
      pip install --no-cache-dir torch==2.9.1+cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu128; \
    else \
      pip install --no-cache-dir torch==2.9.1; \
    fi
```

This block goes **before** the `pip install filter-huggingface-vision` line.

### pyproject.toml change

```toml
# Before
"torch",

# After
"torch>=2.9.1",
```

This ensures the package always requires at least 2.9.1, without locking to a CUDA variant. PyPI-safe.

---

## Step-by-step implementation

1. **`Dockerfile`** (`python:3.13-slim` builder stage):
   - After `RUN pip install --upgrade pip`, add `ARG TARGETPLATFORM`
   - Add the conditional torch pre-install block before `pip install filter-huggingface-vision`

2. **`pyproject.toml`**:
   - Change `"torch"` → `"torch>=2.9.1"` in `[project.dependencies]`

3. No changes needed to:
   - `uv.lock` (not used at Docker build time)
   - `.github/workflows/` (multi-platform `buildx` already sets `TARGETPLATFORM` automatically)
   - `Makefile` (existing `build-image` target already uses `--platform linux/amd64,linux/arm64`)

---

## Verification

```bash
# 1. Build locally
make build-image

# 2. Inspect torch version in the amd64 image
docker run --rm plainsightai/openfilter-huggingface-vision:dev \
  python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Expected: 2.9.1+cu128  12.8

# 3. GPU smoke test on Beast A10 (driver 570.x)
docker run --gpus all plainsightai/openfilter-huggingface-vision:dev \
  python -c "import torch; print(torch.cuda.is_available())"
# Expected: True

# 4. arm64 build — confirm CPU wheel installs cleanly
docker buildx build --platform linux/arm64 --load -t test-arm64 .
docker run --rm test-arm64 \
  python -c "import torch; print(torch.__version__)"
# Expected: 2.9.1  (no +cu128 suffix)
```
