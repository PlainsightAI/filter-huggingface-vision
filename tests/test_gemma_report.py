"""The Gemma prompt is built from counted facts, not from embeddings."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gemma_report import facts_for_model, inmate_side, present, prompt_for


class GemmaPromptTests(unittest.TestCase):
    def test_prompt_keeps_the_five_questions_and_drops_vectors(self):
        report = {
            "gap_s": 20,
            "threshold": 0.45,
            "recordings": ["one.mp4"],
            "calls": [],
            "common_faces": [],
            "visitor_other_inmates": [],
            "inmate_visitor_calls": [],
            "inmate_calls": [],
            "visitor_calls": [],
            "roles_set": False,
            "centroids": {"p1": [0.1, 0.2]},
            "people": [{"id": "p1", "label": "Person 1", "sample": {"src_frame": 68, "roi": [0, 0, 1, 1]}}],
        }
        facts = facts_for_model(report)
        self.assertNotIn("centroids", facts)
        self.assertNotIn("sample", facts["people"][0])
        text = prompt_for(report)
        self.assertIn("common_faces", text)
        self.assertIn("visitor_other_inmates", text)
        self.assertIn("inmate_visitor_calls", text)
        self.assertIn("inmate_calls", text)
        self.assertIn("visitor_calls", text)
        self.assertNotIn("0.1", text)
        self.assertNotIn("src_frame", text)

    def test_json_dump_becomes_five_report_lines(self):
        raw = """```json
{
  "faces_in_more_than_one_recording": "None",
  "visitors_who_share_a_call_with_more_than_one_inmate": [
    {"label": "Person 2", "also_with_other_inmates": false,
     "inmates": [{"label": "Person 1"}]}
  ],
  "inmate_visitor_calls": [
    {"inmate": "Person 1", "visitor": "Person 2", "calls": 1}
  ],
  "inmate_calls": [{"label": "Person 1", "calls": 1}],
  "visitor_calls": [{"label": "Person 2", "calls": 1}]
}
```"""
        text = present(raw)
        self.assertIn("Faces in more than one recording: None.", text)
        self.assertIn("Person 2 does not", text)
        self.assertIn("Person 1 and Person 2 shared 1 call.", text)
        self.assertIn("Inmate calls: Person 1 was in 1 call.", text)
        self.assertIn("Visitor calls: Person 2 was in 1 call.", text)

    def test_inmate_side_reads_one_word(self):
        self.assertEqual(inmate_side("left"), "left")
        self.assertEqual(inmate_side("The inmate is on the right."), "right")
        self.assertEqual(inmate_side("left and right both look possible"), "unknown")
        self.assertEqual(inmate_side("I cannot tell"), "unknown")


if __name__ == "__main__":
    unittest.main()
