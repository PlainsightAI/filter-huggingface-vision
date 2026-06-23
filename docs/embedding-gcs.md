---
title: Exemplar banks from GCS
sidebar_label: Exemplar banks from GCS
sidebar_position: 4
---

The `embedding` backend compares each frame's embedding against a reference
"bank" of exemplar embeddings (an `.npz` file) to emit `min_exemplar_distance`.
`exemplar_embeddings_path` accepts both a **local path** and a **`gs://` URI**,
so the bank can live in a bucket and be read in-pod — no need to bake it into
the image.

```python
exemplar_embeddings_path="gs://YOUR_BUCKET/drift-detection/bank.npz"
```

Loading goes through [`fsspec`](https://filesystem-spec.readthedocs.io/) +
[`gcsfs`](https://gcsfs.readthedocs.io/), giving one code path for local and
`gs://` sources. The `.npz` key resolution is unchanged: `embeddings` →
`arr_0` → first key, shape `(N, dim)`.

## Installing the `gs://` driver

`gcsfs` is an optional extra so installs that never touch object storage stay
lean. The published **Docker image already bundles it**, so docker-compose /
in-pod usage needs nothing extra. For a plain `pip` install, add the `[gcs]`
extra when you want `gs://` support:

```bash
pip install "filter-huggingface-vision[gcs]"
```

Without it, a `gs://` path fails loudly with fsspec's
`ImportError: Please install gcsfs to access Google Storage` (local and
`file://` paths still work without the extra).

A missing bank raises `FileNotFoundError`; other failures (permission denied,
corrupt archive, network error) raise loudly too — as their own error types,
not necessarily `FileNotFoundError`. A bank that loads but is empty or non-2D
raises `ValueError` at init. The loader never falls back to an empty bank,
which would make every frame look like maximal drift.

## Credentials (workload identity / ADC)

No credentials live in config. `gcsfs` auto-discovers them via Application
Default Credentials. On GKE that means **workload identity**: bind the pod's
Kubernetes service account to a Google service account that has read access to
the bank's bucket.

The only IAM the filter needs is object read on the bucket:

```bash
gsutil iam ch \
  serviceAccount:SA_NAME@PROJECT.iam.gserviceaccount.com:roles/storage.objectViewer \
  gs://YOUR_BUCKET
```

(`roles/storage.objectViewer` is sufficient — the filter only reads the bank.)

## Demo fallback (no platform required)

The `gs://` path is optional. To run without any GCS/IAM setup, mount the
`.npz` into the container (or bake it into the image) and pass a local path —
the same code path handles it:

```python
exemplar_embeddings_path="/exemplars/bank.npz"
```
