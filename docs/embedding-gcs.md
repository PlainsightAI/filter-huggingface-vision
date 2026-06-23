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
exemplar_embeddings_path="gs://protege-artifacts-production/drift-detection/bank.npz"
```

Loading goes through [`fsspec`](https://filesystem-spec.readthedocs.io/) +
[`gcsfs`](https://gcsfs.readthedocs.io/), giving one code path for local and
`gs://` sources. The `.npz` key resolution is unchanged: `embeddings` →
`arr_0` → first key, shape `(N, dim)`.

A missing or unreadable bank **raises** (`FileNotFoundError`) — it never falls
back to an empty bank, which would make every frame look like maximal drift.

## Credentials (workload identity / ADC)

No credentials live in config. `gcsfs` auto-discovers them via Application
Default Credentials. On GKE that means **workload identity**: bind the pod's
Kubernetes service account to a Google service account that has read access to
the bank's bucket.

The only IAM the filter needs is object read on the bucket:

```bash
gsutil iam ch \
  serviceAccount:SA_NAME@PROJECT.iam.gserviceaccount.com:roles/storage.objectViewer \
  gs://protege-artifacts-production
```

(`roles/storage.objectViewer` is sufficient — the filter only reads the bank.)

## Demo fallback (no platform required)

The `gs://` path is optional. To run without any GCS/IAM setup, mount the
`.npz` into the container (or bake it into the image) and pass a local path —
the same code path handles it:

```python
exemplar_embeddings_path="/exemplars/bank.npz"
```
