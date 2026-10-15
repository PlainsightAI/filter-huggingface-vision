"""Call counts from clustered face points. No Qdrant."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from face_report import build_report, crop_box


def _point(video, time, vector, roi=None):
    return {
        "vector": {"identity": vector},
        "payload": {
            "video_id": video,
            "pts_s": time,
            "src_frame": int(time),
            "roi": roi or [0, 0, 1, 1],
        },
    }


def _series(video, start, vector, count=4):
    return [_point(video, start + index, vector) for index in range(count)]


class FaceReportTests(unittest.TestCase):
    def test_visitor_seen_with_two_inmates_is_flagged(self):
        visitor = [1.0, 0.0, 0.0]
        inmate_a = [0.0, 1.0, 0.0]
        inmate_b = [0.0, 0.0, 1.0]
        points = (
            _series("file:///calls/one.mp4", 0, visitor)
            + _series("file:///calls/one.mp4", 0.1, inmate_a)
            + _series("file:///calls/two.mp4", 0, visitor)
            + _series("file:///calls/two.mp4", 0.1, inmate_b)
        )
        report = build_report(points, 0.8, {}, min_faces=2)
        self.assertEqual(len(report["calls"]), 2)
        self.assertEqual(len(report["common_faces"]), 1)
        by_label = {}
        roles = {}
        for person in report["people"]:
            if person["recordings_count"] == 2:
                roles[person["id"]] = "visitor"
            else:
                roles[person["id"]] = "inmate"
            by_label[person["label"]] = person
        report = build_report(points, 0.8, roles, min_faces=2)
        self.assertEqual(len(report["visitor_other_inmates"]), 1)
        self.assertTrue(report["visitor_other_inmates"][0]["also_with_other_inmates"])
        self.assertEqual(report["visitor_calls"][0]["calls"], 2)
        self.assertEqual({row["calls"] for row in report["inmate_calls"]}, {1})
        self.assertEqual(len(report["inmate_visitor_calls"]), 2)
        self.assertTrue(all(row["calls"] == 1 for row in report["inmate_visitor_calls"]))
        sample = report["people"][0]["sample"]
        self.assertEqual(len(sample["roi"]), 4)
        self.assertGreaterEqual(sample["src_frame"], 0)

    def test_crop_box_stays_inside_the_frame(self):
        self.assertEqual(crop_box(100, 50, [0.0, 0.0, 0.2, 0.4]), (0, 0, 24, 24))

    def test_left_face_is_the_inmate_and_right_face_is_the_visitor(self):
        inmate = [1.0, 0.0, 0.0]
        visitor = [0.0, 1.0, 0.0]
        other = [0.0, 0.0, 1.0]
        points = [
            _point("file:///calls/one.mp4", 0, inmate, [0.1, 0.2, 0.3, 0.6]),
            _point("file:///calls/one.mp4", 0, visitor, [0.6, 0.2, 0.9, 0.6]),
            _point("file:///calls/one.mp4", 30, inmate, [0.1, 0.2, 0.3, 0.6]),
            _point("file:///calls/one.mp4", 30, visitor, [0.6, 0.2, 0.9, 0.6]),
            _point("file:///calls/one.mp4", 60, other, [0.12, 0.2, 0.28, 0.6]),
            _point("file:///calls/one.mp4", 60, visitor, [0.62, 0.2, 0.88, 0.6]),
        ]
        report = build_report(points, 0.8, {}, min_faces=1)
        by_role = {}
        for person in report["people"]:
            by_role.setdefault(person["role"], []).append(person)
        self.assertEqual(len(by_role["visitor"]), 1)
        self.assertEqual(by_role["visitor"][0]["role_source"], "right")
        self.assertEqual(len(by_role["inmate"]), 2)
        pair = report["inmate_visitor_calls"]
        repeated = [row for row in pair if row["calls"] == 2]
        self.assertEqual(len(repeated), 1)
        self.assertTrue(report["visitor_other_inmates"][0]["also_with_other_inmates"])
        flipped = build_report(points, 0.8, {}, min_faces=1, flip=True)
        self.assertEqual(
            {person["role"] for person in flipped["people"] if person["role_source"] == "right"},
            {"inmate"},
        )


if __name__ == "__main__":
    unittest.main()
