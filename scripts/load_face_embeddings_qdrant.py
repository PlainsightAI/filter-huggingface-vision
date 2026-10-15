"""Load recorder face embeddings into the local Qdrant.

One point per face. Named vectors stay in separate spaces:
``identity`` is the L2-normalized ArcFace vector, ``appearance`` is the
L2-normalized DINOv2 vector. Both use dot product, which is cosine on
unit vectors. Existing collections are left untouched.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import uuid

IDENTITY_MODEL = "minchul/cvlface_arcface_ir101_webface4m"
APPEARANCE_MODEL = "facebook/dinov2-small"
COLLECTION = "fdc_eval_faces"
NAMESPACE = uuid.UUID("8f1b6c3e-4a27-5d90-9c1e-fdc0face0001")


def _request(url: str, method: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} -> {exc.code}: {detail}") from exc


def ensure_collection(base: str) -> None:
    try:
        _request(f"{base}/collections/{COLLECTION}", "GET")
        return
    except RuntimeError as exc:
        if "404" not in str(exc):
            raise
    _request(
        f"{base}/collections/{COLLECTION}",
        "PUT",
        {
            "vectors": {
                "identity": {"size": 512, "distance": "Dot"},
                "appearance": {"size": 384, "distance": "Dot"},
            }
        },
    )
    for field in ("video_id", "src_frame"):
        _request(
            f"{base}/collections/{COLLECTION}/index?wait=true",
            "PUT",
            {"field_name": field, "field_schema": "keyword" if field == "video_id" else "integer"},
        )


def _vector(embeddings, model_id: str, size: int):
    for item in embeddings:
        if item.get("model") != model_id:
            continue
        vector = item.get("vector")
        if isinstance(vector, list) and len(vector) == size:
            return vector
    return None


def points_from_line(line: str):
    row = json.loads(line)
    main = row.get("main") or {}
    meta = main.get("meta") or {}
    video_id = meta.get("src")
    src_frame = meta.get("src_frame")
    found = []
    for index, detection in enumerate(meta.get("detections") or []):
        embeddings = detection.get("embeddings") or []
        identity = _vector(embeddings, IDENTITY_MODEL, 512)
        appearance = _vector(embeddings, APPEARANCE_MODEL, 384)
        if identity is None or appearance is None:
            continue
        rois = detection.get("rois") or []
        point_id = str(uuid.uuid5(NAMESPACE, f"{video_id}|{src_frame}|{index}"))
        found.append(
            {
                "id": point_id,
                "vector": {"identity": identity, "appearance": appearance},
                "payload": {
                    "video_id": video_id,
                    "src_frame": src_frame,
                    "pts_s": meta.get("pts_s")
                    if meta.get("pts_s") is not None
                    else meta.get("src_seconds"),
                    "ts": meta.get("ts"),
                    "class": detection.get("class"),
                    "roi": rois[0] if rois else None,
                    "detection_index": index,
                    "identity_model": IDENTITY_MODEL,
                    "appearance_model": APPEARANCE_MODEL,
                },
            }
        )
    return found


def load(path: str, base: str, batch_size: int) -> int:
    ensure_collection(base)
    batch = []
    written = 0

    def flush():
        nonlocal written
        if not batch:
            return
        _request(
            f"{base}/collections/{COLLECTION}/points?wait=true",
            "PUT",
            {"points": batch},
        )
        written += len(batch)
        batch.clear()

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            batch.extend(points_from_line(line))
            if len(batch) >= batch_size:
                flush()
    flush()
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default="/home/trodrigues138/filter-huggingface-vision/output/fdc-eval-15min.jsonl",
    )
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    count = load(args.jsonl, args.url.rstrip("/"), args.batch_size)
    info = _request(f"{args.url.rstrip('/')}/collections/{COLLECTION}", "GET")
    print(f"upserted {count} points; collection status {info['result']['points_count']} points")


if __name__ == "__main__":
    main()
