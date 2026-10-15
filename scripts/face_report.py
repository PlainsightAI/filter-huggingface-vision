"""Group saved face points into people and the calls they share.

One recording can hold several calls. A new call starts when the same
recording goes 20 seconds with no face. Two people shared a call when both
are present in that stretch. Inmate and visitor are marks from the UI.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

COLLECTION = "fdc_eval_faces"
CALL_GAP_S = 20.0


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


def video_name(video_id) -> str:
    text = str(video_id or "")
    if text.startswith("file://"):
        return Path(text[7:]).name
    return text or "recording"


def scroll_faces(base: str) -> list[dict]:
    points = []
    offset = None
    while True:
        body = {
            "limit": 256,
            "with_payload": ["video_id", "src_frame", "pts_s", "roi"],
            "with_vector": ["identity"],
        }
        if offset is not None:
            body["offset"] = offset
        page = _request(f"{base}/collections/{COLLECTION}/points/scroll", "POST", body)
        result = page["result"]
        points.extend(result.get("points") or [])
        offset = result.get("next_page_offset")
        if offset is None:
            break
    points.sort(
        key=lambda point: (
            str((point.get("payload") or {}).get("video_id") or ""),
            float((point.get("payload") or {}).get("pts_s") or 0.0),
            int((point.get("payload") or {}).get("src_frame") or 0),
        )
    )
    return points


def _unit(vector) -> np.ndarray:
    row = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(row))
    if norm <= 0:
        return row
    return row / norm


def _identity(point: dict):
    raw = point.get("vector") or {}
    if isinstance(raw, dict):
        return raw.get("identity")
    return raw


def cluster_faces(points: list[dict], threshold: float) -> list[dict]:
    """Greedy clusters, then merge centroids that still match."""
    people = []
    for point in points:
        vector = _identity(point)
        if not vector:
            continue
        sample = _unit(vector)
        payload = point.get("payload") or {}
        video_id = payload.get("video_id")
        if people:
            scores = np.stack([person["centroid"] for person in people]) @ sample
            index = int(np.argmax(scores))
            if float(scores[index]) >= threshold:
                person = people[index]
                person["count"] += 1
                person["sum"] += sample
                person["centroid"] = _unit(person["sum"])
                person["videos"].add(video_id)
                continue
        people.append(
            {
                "centroid": sample.copy(),
                "sum": sample.copy(),
                "count": 1,
                "videos": {video_id},
            }
        )
    return _merge(people, threshold)


def _merge(people: list[dict], threshold: float) -> list[dict]:
    people = [
        dict(person, videos=set(person["videos"]), sum=person["sum"].copy(), centroid=person["centroid"].copy())
        for person in people
    ]
    while len(people) > 1:
        stacked = np.stack([person["centroid"] for person in people])
        scores = stacked @ stacked.T
        np.fill_diagonal(scores, -1)
        left, right = np.unravel_index(int(np.argmax(scores)), scores.shape)
        if float(scores[left, right]) < threshold:
            break
        keep, drop = people[int(left)], people[int(right)]
        keep["sum"] = keep["sum"] + drop["sum"]
        keep["count"] += drop["count"]
        keep["centroid"] = _unit(keep["sum"])
        keep["videos"] |= drop["videos"]
        people.pop(int(right))
    return people


def _side_x(points: list[dict], people: list[dict], threshold: float) -> dict[int, float]:
    """Median horizontal center of each person's face. 0 is the left edge."""
    if not people:
        return {}
    centroids = np.stack([person["centroid"] for person in people])
    centers: dict[int, list[float]] = {}
    for point in points:
        vector = _identity(point)
        if not vector:
            continue
        scores = centroids @ _unit(vector)
        index = int(np.argmax(scores))
        if float(scores[index]) < threshold:
            continue
        roi = (point.get("payload") or {}).get("roi")
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            continue
        centers.setdefault(index, []).append((float(roi[0]) + float(roi[2])) / 2)
    return {index: float(np.median(values)) for index, values in centers.items()}


def _role_from_side(x: float | None, flip: bool) -> tuple[str, str]:
    """Left side of the frame is the inmate. Right side is the visitor.

    A visitation frame puts one person on each side. The two labels swap
    together when ``flip`` is set, because the booth can face the other way.
    """
    if x is None or abs(x - 0.5) < 1e-6:
        return "", ""
    on_left = x < 0.5
    inmate_on_left = not flip
    role = "inmate" if on_left == inmate_on_left else "visitor"
    return role, "left" if on_left else "right"


def _best_samples(points: list[dict], people: list[dict], threshold: float) -> dict[int, dict]:
    """The detection closest to each person, so the UI can show that face."""
    if not people:
        return {}
    centroids = np.stack([person["centroid"] for person in people])
    best: dict[int, dict] = {}
    for point in points:
        vector = _identity(point)
        if not vector:
            continue
        scores = centroids @ _unit(vector)
        index = int(np.argmax(scores))
        score = float(scores[index])
        if score < threshold:
            continue
        current = best.get(index)
        if current is not None and current["score"] >= score:
            continue
        payload = point.get("payload") or {}
        roi = payload.get("roi")
        best[index] = {
            "score": score,
            "src_frame": int(payload.get("src_frame") or 0),
            "pts_s": float(payload.get("pts_s") or 0.0),
            "roi": list(roi) if isinstance(roi, (list, tuple)) and len(roi) == 4 else None,
        }
    return best


def crop_box(width: int, height: int, roi: list, pad: float = 0.2) -> tuple[int, int, int, int]:
    """Pixel box for a normalized [xmin, ymin, xmax, ymax] face, with a little context."""
    xmin, ymin, xmax, ymax = (float(value) for value in roi)
    face_w = max(xmax - xmin, 0.01)
    face_h = max(ymax - ymin, 0.01)
    xmin -= face_w * pad
    xmax += face_w * pad
    ymin -= face_h * pad
    ymax += face_h * pad
    x0 = max(0, min(width - 1, int(xmin * width)))
    y0 = max(0, min(height - 1, int(ymin * height)))
    x1 = max(x0 + 1, min(width, int(xmax * width)))
    y1 = max(y0 + 1, min(height, int(ymax * height)))
    return x0, y0, x1, y1


def _events(points: list[dict], people: list[dict], threshold: float) -> list[tuple]:
    if not people:
        return []
    centroids = np.stack([person["centroid"] for person in people])
    events = []
    for point in points:
        vector = _identity(point)
        if not vector:
            continue
        scores = centroids @ _unit(vector)
        index = int(np.argmax(scores))
        if float(scores[index]) < threshold:
            continue
        payload = point.get("payload") or {}
        events.append(
            (
                str(payload.get("video_id") or ""),
                float(payload.get("pts_s") or 0.0),
                index,
            )
        )
    events.sort()
    return events


def split_calls(events: list[tuple], gap_s: float = CALL_GAP_S, min_faces: int = 80) -> list[dict]:
    if not events:
        return []
    groups = [[events[0]]]
    for event in events[1:]:
        previous = groups[-1][-1]
        if event[0] != previous[0] or event[1] - previous[1] > gap_s:
            groups.append([event])
            continue
        groups[-1].append(event)
    calls = []
    for index, group in enumerate(groups, start=1):
        counts: dict[int, int] = {}
        for _video, _time, person_index in group:
            counts[person_index] = counts.get(person_index, 0) + 1
        # A dense recording ignores one-off faces. A short deduped run has
        # fewer rows than min_faces, so every matched face in that stretch counts.
        if len(group) < min_faces:
            bar = 1
        else:
            bar = max(min_faces, int(0.05 * len(group)))
        present = [person for person, count in counts.items() if count >= bar]
        calls.append(
            {
                "id": f"c{index}",
                "video_id": group[0][0],
                "video": video_name(group[0][0]),
                "start_s": group[0][1],
                "end_s": group[-1][1],
                "faces": len(group),
                "present": present,
                "counts": counts,
            }
        )
    return calls


def _clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def build_report(
    points: list[dict],
    threshold: float,
    roles: dict[str, str],
    min_faces: int = 80,
    flip: bool = False,
) -> dict:
    people = cluster_faces(points, threshold)
    samples = _best_samples(points, people, threshold)
    sides = _side_x(points, people, threshold)
    events = _events(points, people, threshold)
    calls = split_calls(events, min_faces=min_faces)
    order = sorted(
        range(len(people)),
        key=lambda index: (
            min((call["start_s"] for call in calls if index in call["present"]), default=1e12),
            -people[index]["count"],
        ),
    )
    labels = {}
    catalog = []
    for rank, index in enumerate(order, start=1):
        if not any(index in call["present"] for call in calls):
            continue
        person_id = f"p{rank}"
        labels[index] = person_id
        manual = person_id in roles
        role = roles.get(person_id) or ""
        if role not in ("inmate", "visitor"):
            role = ""
        source = "marked" if manual and role else ""
        if not manual:
            role, source = _role_from_side(sides.get(index), flip)
        member_of = [call for call in calls if index in call["present"]]
        recordings = sorted({call["video"] for call in member_of})
        catalog.append(
            {
                "id": person_id,
                "label": f"Person {rank}",
                "role": role,
                "role_source": source,
                "side_x": sides.get(index),
                "appearances": people[index]["count"],
                "calls": len(member_of),
                "recordings_count": len(recordings),
                "recordings": recordings,
                "call_ids": [call["id"] for call in member_of],
                "common_across_recordings": len(recordings) > 1,
                "common_across_calls": len(member_of) > 1,
                "sample": {
                    "src_frame": (samples.get(index) or {}).get("src_frame", 0),
                    "pts_s": (samples.get(index) or {}).get("pts_s", 0.0),
                    "roi": (samples.get(index) or {}).get("roi"),
                },
                "centroid": people[index]["centroid"],
            }
        )
    by_id = {person["id"]: person for person in catalog}

    def companions(person_id: str) -> list[dict]:
        person = by_id[person_id]
        found = []
        for other in catalog:
            if other["id"] == person_id:
                continue
            shared = sorted(set(person["call_ids"]) & set(other["call_ids"]))
            if not shared:
                continue
            found.append(
                {
                    "id": other["id"],
                    "label": other["label"],
                    "role": other["role"],
                    "calls": len(shared),
                    "call_ids": shared,
                }
            )
        return found

    for person in catalog:
        person["with"] = companions(person["id"])

    public_calls = []
    for call in calls:
        members = []
        for index in call["present"]:
            person_id = labels.get(index)
            if person_id is None:
                continue
            person = by_id[person_id]
            members.append(
                {
                    "id": person["id"],
                    "label": person["label"],
                    "role": person["role"],
                    "appearances": call["counts"].get(index, 0),
                }
            )
        members.sort(key=lambda item: -item["appearances"])
        public_calls.append(
            {
                "id": call["id"],
                "video": call["video"],
                "start_s": call["start_s"],
                "end_s": call["end_s"],
                "start": _clock(call["start_s"]),
                "end": _clock(call["end_s"]),
                "people": members,
            }
        )

    inmates = [person for person in catalog if person["role"] == "inmate"]
    visitors = [person for person in catalog if person["role"] == "visitor"]
    visitor_rows = []
    for visitor in visitors:
        met = [item for item in visitor["with"] if item["role"] == "inmate"]
        visitor_rows.append(
            {
                "id": visitor["id"],
                "label": visitor["label"],
                "calls": visitor["calls"],
                "inmates": met,
                "also_with_other_inmates": len(met) > 1,
            }
        )
    pair_rows = []
    seen = set()
    for visitor in visitors:
        for inmate in visitor["with"]:
            if inmate["role"] != "inmate":
                continue
            key = (inmate["id"], visitor["id"])
            if key in seen:
                continue
            seen.add(key)
            pair_rows.append(
                {
                    "inmate": by_id[inmate["id"]]["label"],
                    "visitor": visitor["label"],
                    "calls": inmate["calls"],
                    "call_ids": inmate["call_ids"],
                }
            )

    published = []
    centroids = {}
    for person in catalog:
        row = dict(person)
        centroids[person["id"]] = row.pop("centroid")
        published.append(row)
    return {
        "gap_s": CALL_GAP_S,
        "threshold": threshold,
        "role_rule": "right" if flip else "left",
        "recordings": sorted({call["video"] for call in public_calls}),
        "calls": public_calls,
        "people": published,
        "common_faces": [
            {
                "id": person["id"],
                "label": person["label"],
                "calls": person["calls"],
                "recordings": person["recordings"],
            }
            for person in published
            if person["common_across_recordings"]
        ],
        "visitor_other_inmates": visitor_rows,
        "inmate_visitor_calls": pair_rows,
        "inmate_calls": [
            {"id": person["id"], "label": person["label"], "calls": person["calls"]}
            for person in inmates
        ],
        "visitor_calls": [
            {"id": person["id"], "label": person["label"], "calls": person["calls"]}
            for person in visitors
        ],
        "roles_set": bool(inmates or visitors),
        "centroids": centroids,
    }


def match_person(centroids: dict, vector: list[float], threshold: float) -> dict | None:
    """Best person id for a query vector. ``centroids`` maps id to a unit vector."""
    if not centroids:
        return None
    sample = _unit(vector)
    best_id = None
    best_score = threshold
    for person_id, centroid in centroids.items():
        score = float(np.dot(centroid, sample))
        if score >= best_score:
            best_score = score
            best_id = person_id
    if best_id is None:
        return None
    return {"id": best_id, "score": best_score}
