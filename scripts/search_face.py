"""Search the saved face points with a photo.

The photo is aligned and embedded with the same ArcFace used at ingest.
Dot product on that 512-d vector is the identity score. Hits on one video
are grouped into time ranges.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

from PIL import Image

from filter_huggingface_vision.arcface import ArcFaceEmbedder
from filter_huggingface_vision.detection_embeddings import l2_normalize
from filter_huggingface_vision.face_align import FaceAligner, warp_face

IDENTITY_MODEL = "minchul/cvlface_arcface_ir101_webface4m"
COLLECTION = "fdc_eval_faces"

DEFAULT_URL = "http://127.0.0.1:6333"


def _request(url: str, payload: dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} -> {exc.code}: {detail}") from exc


def search(url: str, vector: list[float], limit: int, threshold: float):
    body = {
        "query": vector,
        "using": "identity",
        "limit": limit,
        "with_payload": True,
        "score_threshold": threshold,
    }
    result = _request(f"{url}/collections/{COLLECTION}/points/query", body)
    return result["result"]["points"]


def segments(hits, gap_seconds: float):
    ordered = sorted(
        hits,
        key=lambda hit: (
            str(hit["payload"].get("video_id")),
            float(hit["payload"].get("pts_s") or 0.0),
            int(hit["payload"].get("src_frame") or 0),
        ),
    )
    groups = []
    for hit in ordered:
        payload = hit["payload"]
        video_id = payload.get("video_id")
        start = float(payload.get("pts_s") or 0.0)
        frame = payload.get("src_frame")
        score = float(hit["score"])
        if (
            not groups
            or groups[-1]["video_id"] != video_id
            or start - groups[-1]["end_s"] > gap_seconds
        ):
            groups.append(
                {
                    "video_id": video_id,
                    "start_s": start,
                    "end_s": start,
                    "start_frame": frame,
                    "end_frame": frame,
                    "hits": 1,
                    "best": score,
                }
            )
            continue
        group = groups[-1]
        group["end_s"] = start
        group["end_frame"] = frame
        group["hits"] += 1
        group["best"] = max(group["best"], score)
    return groups


def clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


class IdentitySearcher:
    """Keeps the aligner and ArcFace loaded for repeated photo queries."""

    def __init__(self, device: str = "cpu", url: str = DEFAULT_URL):
        self.url = url.rstrip("/")
        self.aligner = FaceAligner().load(device)
        self.embedder = ArcFaceEmbedder().load(
            {"model_id": IDENTITY_MODEL, "revision": "main", "device": device}
        )

    def embed(self, image: Image.Image):
        found = []
        for box, landmarks, score in self.aligner._faces(image):
            aligned = warp_face(image, landmarks)
            if aligned is None:
                continue
            vector = l2_normalize(self.embedder.vector(aligned))
            found.append(
                {
                    "box": [float(v) for v in box],
                    "score": float(score),
                    "vector": vector,
                    "aligned": aligned,
                }
            )
        return found

    def shutdown(self):
        self.aligner.shutdown()
        self.embedder.shutdown()


def query_faces(image: Image.Image, device: str):
    searcher = IdentitySearcher(device)
    try:
        return searcher.embed(image)
    finally:
        searcher.shutdown()


def report(faces, url: str, limit: int, threshold: float, gap_seconds: float, face_index: int) -> int:
    if not faces:
        print("No face found in the image.")
        return 1
    if face_index:
        if face_index < 1 or face_index > len(faces):
            print(f"The image has {len(faces)} face(s); --face {face_index} does not exist.")
            return 1
        chosen = [(face_index, faces[face_index - 1])]
    else:
        chosen = list(enumerate(faces, start=1))
    present = False
    for index, face in chosen:
        hits = search(url, face["vector"], limit, threshold)
        best = max((float(hit["score"]) for hit in hits), default=0.0)
        lowest = min((float(hit["score"]) for hit in hits), default=0.0)
        print(
            f"face {index}: detection {face['score']:.2f}, "
            f"best score {best:.3f}, threshold {threshold:.3f}, matches {len(hits)}"
        )
        if len(hits) >= limit:
            print(f"  result cut at the limit {limit}; the span may continue")
        if not hits:
            print("  present: no")
            continue
        present = True
        print(f"  present: yes (lowest score in this result {lowest:.3f})")
        for group in segments(hits, gap_seconds):
            print(
                f"  {group['video_id']}  "
                f"{clock(group['start_s'])}–{clock(group['end_s'])}  "
                f"frames {group['start_frame']}–{group['end_frame']}  "
                f"matches {group['hits']}  best {group['best']:.3f}"
            )
    return 0 if present else 2


def main():
    parser = argparse.ArgumentParser(description="Find a face photo in the saved Qdrant points.")
    parser.add_argument("image", help="Photo or frame that contains the face")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--gap-seconds", type=float, default=1.5)
    parser.add_argument("--face", type=int, default=0, help="1-based face in the photo; 0 searches every face")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    image = Image.open(args.image)
    searcher = IdentitySearcher(args.device, args.url)
    faces = searcher.embed(image)
    raise SystemExit(
        report(
            faces,
            args.url.rstrip("/"),
            args.limit,
            args.threshold,
            args.gap_seconds,
            args.face,
        )
    )


if __name__ == "__main__":
    main()
