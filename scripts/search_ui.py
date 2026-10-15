"""Local page to search the saved face points with a photo.

    uv run python scripts/search_ui.py

Then open http://127.0.0.1:8096
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

from face_report import build_report, crop_box, match_person, scroll_faces
from gemma_report import judge_inmate_side, write_report
from search_face import COLLECTION, DEFAULT_URL, IdentitySearcher, clock, search, segments

HOST = "0.0.0.0"
PORT = 8096
PAGE = Path(__file__).with_name("search_ui.html")
CLIP = Path("/home/leandrobmarinho/internal-fdc/pipelines/data/prison-visit-10473644.mp4")
LIMIT = 5000
GAP_SECONDS = 1.5
MAX_BYTES = 12_000_000

_searcher: IdentitySearcher | None = None
_lock = threading.Lock()
_duration_s: float | None = None
_points: list | None = None
_roles: dict[str, str] = {}
_reports: dict = {}
_flip = False
_judgment: dict | None = None
_judge_lock = threading.Lock()
_crops: dict = {}
_video_lock = threading.Lock()


def clip_duration() -> float | None:
    if not CLIP.is_file():
        return None
    capture = cv2.VideoCapture(str(CLIP))
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    capture.release()
    if frames <= 0 or fps <= 0:
        return None
    return frames / fps


def video_name(video_id) -> str:
    text = str(video_id or "")
    if text.startswith("file://"):
        return Path(text[7:]).name
    return text or "video"


def _jpeg_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def prepare_image(raw: bytes) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    image = image.convert("RGB")
    image.thumbnail((1600, 1600))
    return image


def _stored_points() -> list:
    global _points
    if _points is None:
        _points = scroll_faces(DEFAULT_URL)
    return _points


def call_report(threshold: float) -> dict:
    key = (round(threshold, 3), tuple(sorted(_roles.items())), _flip)
    cached = _reports.get(key)
    if cached is None:
        cached = build_report(_stored_points(), threshold, _roles, flip=_flip)
        _reports[key] = cached
    public = dict(cached)
    public.pop("centroids", None)
    public["side_judgment"] = _judgment
    return public


def sample_visit_frame() -> Image.Image | None:
    """One frame that shows a person on the left and a person on the right."""
    grouped: dict[int, list] = {}
    for point in _stored_points():
        payload = point.get("payload") or {}
        roi = payload.get("roi")
        frame_index = payload.get("src_frame")
        if not isinstance(roi, (list, tuple)) or len(roi) != 4 or frame_index is None:
            continue
        grouped.setdefault(int(frame_index), []).append(roi)
    chosen = None
    for frame_index in sorted(grouped):
        centers = [(float(roi[0]) + float(roi[2])) / 2 for roi in grouped[frame_index]]
        if any(center < 0.45 for center in centers) and any(center > 0.55 for center in centers):
            chosen = frame_index
            break
    if chosen is None or not CLIP.is_file():
        return None
    with _video_lock:
        capture = cv2.VideoCapture(str(CLIP))
        capture.set(cv2.CAP_PROP_POS_FRAMES, chosen)
        ok, frame = capture.read()
        capture.release()
    if not ok or frame is None:
        return None
    height, width = frame.shape[:2]
    cv2.putText(frame, "LEFT", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT", (max(16, width - 180), 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.thumbnail((960, 960))
    return image


def ensure_side_judgment() -> None:
    """Look at one visit frame once, before the report labels inmate and visitor."""
    global _judgment
    if _judgment is not None:
        return
    with _judge_lock:
        if _judgment is not None:
            return
        try:
            apply_side_judgment()
        except Exception as exc:
            _judgment = {"inmate_side": "unknown", "reply": str(exc)}


def apply_side_judgment() -> dict:
    """Let Gemma look at one visit frame and set which side is the inmate."""
    global _flip, _judgment
    image = sample_visit_frame()
    if image is None:
        raise RuntimeError("No frame shows a person on each side.")
    result = judge_inmate_side(image)
    side = result["inmate_side"]
    with _lock:
        if side in ("left", "right"):
            _flip = side == "right"
            _roles.clear()
            _reports.clear()
        _judgment = {"inmate_side": side, "reply": result["reply"]}
    return {"inmate_side": side, "reply": result["reply"], "flip": _flip}


def face_jpeg(person_id: str, threshold: float) -> bytes | None:
    """Crop the saved frame that best represents this person."""
    with _lock:
        report = call_report(threshold)
    person = next((item for item in report["people"] if item["id"] == person_id), None)
    sample = (person or {}).get("sample") or {}
    roi = sample.get("roi")
    if person is None or not roi or not CLIP.is_file():
        return None
    frame_index = int(sample.get("src_frame") or 0)
    key = (frame_index, tuple(round(float(value), 4) for value in roi))
    cached = _crops.get(key)
    if cached is not None:
        return cached
    jpeg = _read_crop(frame_index, roi)
    if jpeg is not None:
        _crops[key] = jpeg
    return jpeg


def _read_crop(src_frame: int, roi: list) -> bytes | None:
    with _video_lock:
        capture = cv2.VideoCapture(str(CLIP))
        capture.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
        ok, frame = capture.read()
        capture.release()
    if not ok or frame is None:
        return None
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = crop_box(width, height, roi)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return encoded.tobytes()


def match_photo(image: Image.Image, threshold: float):
    with _lock:
        faces = _searcher.embed(image)
        call_report(threshold)
        report = _reports[(round(threshold, 3), tuple(sorted(_roles.items())))]
        centroids = report.get("centroids") or {}
        people = {person["id"]: person for person in report.get("people") or []}
        matched = []
        for index, face in enumerate(faces, start=1):
            hits = search(_searcher.url, face["vector"], LIMIT, threshold)
            groups = segments(hits, GAP_SECONDS)
            best = max((float(hit["score"]) for hit in hits), default=0.0)
            person = match_person(centroids, face["vector"], threshold)
            if person is not None:
                known = people.get(person["id"]) or {}
                person = {
                    "id": person["id"],
                    "label": known.get("label") or person["id"],
                    "role": known.get("role") or "",
                    "score": person["score"],
                    "calls": known.get("calls") or 0,
                    "common_across_calls": bool(known.get("common_across_calls")),
                    "common_across_recordings": bool(known.get("common_across_recordings")),
                }
            matched.append(
                {
                    "index": index,
                    "box": face["box"],
                    "detection_score": face["score"],
                    "crop": _jpeg_data_url(face["aligned"]),
                    "present": bool(hits),
                    "best": best,
                    "count": len(hits),
                    "truncated": len(hits) >= LIMIT,
                    "person": person,
                    "segments": [
                        {
                            "video_id": group["video_id"],
                            "video": video_name(group["video_id"]),
                            "start_s": group["start_s"],
                            "end_s": group["end_s"],
                            "start": clock(group["start_s"]),
                            "end": clock(group["end_s"]),
                            "start_frame": group["start_frame"],
                            "end_frame": group["end_frame"],
                            "hits": group["hits"],
                            "best": group["best"],
                        }
                        for group in groups
                    ],
                }
            )
    return {
        "faces": matched,
        "preview": _jpeg_data_url(image),
        "duration_s": _duration_s,
        "clip": CLIP.is_file(),
        "collection": COLLECTION,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/clip":
            self._clip()
            return
        if path == "/api/health":
            self._json(200, {"ok": True, "collection": COLLECTION})
            return
        if path == "/api/report":
            self._report()
            return
        if path == "/api/gemma-report":
            self._gemma_report()
            return
        if path.startswith("/api/face/"):
            self._face(path.removeprefix("/api/face/"))
            return
        self._json(404, {"error": "Not found."})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/role":
            self._set_role()
            return
        if path == "/api/swap-sides":
            self._swap_sides()
            return
        if path == "/api/judge-sides":
            self._judge_sides()
            return
        if path != "/api/search":
            self._json(404, {"error": "Not found."})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BYTES * 2:
            self._json(400, {"error": "Image is empty or too large."})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            raw = base64.b64decode(payload["image"], validate=True)
            threshold = float(payload.get("threshold", 0.45))
        except (KeyError, ValueError, json.JSONDecodeError):
            self._json(400, {"error": "Send an image and a threshold."})
            return
        if not 0 < threshold < 1:
            self._json(400, {"error": "The threshold must be between 0 and 1."})
            return
        if not raw or len(raw) > MAX_BYTES:
            self._json(400, {"error": "Image is empty or too large."})
            return
        try:
            image = prepare_image(raw)
            result = match_photo(image, threshold)
        except UnidentifiedImageError:
            self._json(400, {"error": "That file is not an image."})
            return
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        self._json(200, result)

    def _report(self):
        parsed = urlparse(self.path)
        raw = "0.45"
        if parsed.query:
            for part in parsed.query.split("&"):
                if part.startswith("threshold="):
                    raw = part.split("=", 1)[1]
        try:
            threshold = float(raw)
        except ValueError:
            self._json(400, {"error": "The threshold must be a number."})
            return
        if not 0 < threshold < 1:
            self._json(400, {"error": "The threshold must be between 0 and 1."})
            return
        try:
            ensure_side_judgment()
            with _lock:
                payload = call_report(threshold)
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        self._json(200, payload)

    def _face(self, person_id: str):
        parsed = urlparse(self.path)
        raw = "0.45"
        if parsed.query:
            for part in parsed.query.split("&"):
                if part.startswith("threshold="):
                    raw = part.split("=", 1)[1]
        try:
            threshold = float(raw)
        except ValueError:
            self._json(400, {"error": "The threshold must be a number."})
            return
        if not 0 < threshold < 1:
            self._json(400, {"error": "The threshold must be between 0 and 1."})
            return
        try:
            jpeg = face_jpeg(person_id, threshold)
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        if jpeg is None:
            self._json(404, {"error": "No face crop for that person."})
            return
        self._send(200, jpeg, "image/jpeg")

    def _gemma_report(self):
        parsed = urlparse(self.path)
        raw = "0.45"
        if parsed.query:
            for part in parsed.query.split("&"):
                if part.startswith("threshold="):
                    raw = part.split("=", 1)[1]
        try:
            threshold = float(raw)
        except ValueError:
            self._json(400, {"error": "The threshold must be a number."})
            return
        if not 0 < threshold < 1:
            self._json(400, {"error": "The threshold must be between 0 and 1."})
            return
        try:
            with _lock:
                facts = call_report(threshold)
            narrative = write_report(facts)
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        self._json(200, {"model": "google/gemma-4-E2B-it", "report": narrative})

    def _set_role(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            person_id = str(payload["id"])
            role = str(payload.get("role") or "")
        except (KeyError, ValueError, json.JSONDecodeError):
            self._json(400, {"error": "Send a person id and a role."})
            return
        if role not in ("", "inmate", "visitor"):
            self._json(400, {"error": "Role must be inmate, visitor, or empty."})
            return
        with _lock:
            if role:
                _roles[person_id] = role
            else:
                _roles.pop(person_id, None)
            _reports.clear()
        self._json(200, {"ok": True})

    def _judge_sides(self):
        try:
            result = apply_side_judgment()
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        self._json(200, result)

    def _swap_sides(self):
        global _flip
        with _lock:
            _flip = not _flip
            _roles.clear()
            _reports.clear()
        self._json(200, {"flip": _flip})

    def _clip(self):
        if not CLIP.is_file():
            self._json(404, {"error": "The video is not on this server."})
            return
        size = CLIP.stat().st_size
        content_type = mimetypes.guess_type(CLIP.name)[0] or "video/mp4"
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match is None:
                self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                return
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            elif match.group(2):
                start = max(0, size - int(match.group(2)))
            end = min(end, size - 1)
            if start > end or start >= size:
                self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                return
            status = 206
        length = end - start + 1
        if status == 206:
            length = min(length, 1024 * 1024)
            end = start + length - 1
        try:
            with CLIP.open("rb") as handle:
                handle.seek(start)
                chunk = handle.read(length)
        except OSError as exc:
            self._json(500, {"error": str(exc)})
            return
        extra = {"Accept-Ranges": "bytes"}
        if status == 206:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Cache-Control", "no-store")
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return


def main():
    global _searcher, _duration_s
    print("Loading the aligner and ArcFace…", flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        _searcher = IdentitySearcher("cpu", DEFAULT_URL)
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        print("Local cache is incomplete; downloading the models.", flush=True)
        _searcher = IdentitySearcher("cpu", DEFAULT_URL)
    _duration_s = clip_duration()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Face search at http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        _searcher.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
