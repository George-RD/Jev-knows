"""QA harness majority-vote sampling; the reader and judge are stubbed, no network."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import longmemeval_qa as qa  # noqa: E402

ITEM = {
    "question_id": "q1",
    "question_type": "multi-session",
    "question": "How many?",
    "question_date": "2023/05/30 (Tue) 10:00",
    "answer": 3,
}


def scripted(verdicts):
    """A chat stub: the reader answers "a<n>", the judge says the next verdict."""
    calls = iter(verdicts)
    count = iter(range(100))

    def chat(model, prompt):
        return next(calls) if model == "judge" else f"a{next(count)}"

    return chat


class SamplesTests(unittest.TestCase):
    def setUp(self):
        qa._CONFIG.update({"reader": "reader", "judge": "judge"})

    def test_majority_decides(self):
        with patch.object(qa, "chat", scripted(["no", "yes", "yes"])):
            out = qa.answer(ITEM, "notes", samples=3)
        self.assertTrue(out["correct"])
        self.assertEqual(out["votes"], 2)
        self.assertEqual([s["correct"] for s in out["samples"]], [False, True, True])
        self.assertEqual((out["hypothesis"], out["judge_raw"]), ("a0", "no"))

    def test_minority_is_wrong_and_even_ties_are_wrong(self):
        with patch.object(qa, "chat", scripted(["yes", "no", "no"])):
            self.assertFalse(qa.answer(ITEM, "notes", samples=3)["correct"])
        with patch.object(qa, "chat", scripted(["yes", "no"])):
            self.assertFalse(qa.answer(ITEM, "notes", samples=2)["correct"])

    def test_one_sample_keeps_the_old_row_shape(self):
        with patch.object(qa, "chat", scripted(["yes"])):
            out = qa.answer(ITEM, "notes", samples=1)
        self.assertEqual(out, {"hypothesis": "a0", "judge_raw": "yes", "correct": True})

    def test_a_failed_sample_makes_an_error_row(self):
        def chat(model, prompt):
            if model == "judge":
                raise TimeoutError("slow")
            return "a"

        with patch.object(qa, "chat", chat):
            out = qa.answer(ITEM, "notes", samples=3)
        self.assertNotIn("correct", out)
        self.assertIn("TimeoutError", out["error"])

    def test_summary_reports_per_sample_accuracy(self):
        rows = [
            {
                "abstention": False,
                "question_type": "t",
                "correct": True,
                "samples": [{"correct": True}, {"correct": True}, {"correct": False}],
            },
            {"abstention": False, "question_type": "t", "correct": True},  # one-sample row
            {"abstention": False, "question_type": "t", "error": "x"},
        ]
        summary = qa.summarize(rows)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["accuracy_per_sample"], 0.75)
        self.assertEqual(summary["by_type"]["t"]["accuracy_per_sample"], 0.75)

    def test_resume_treats_old_reports_as_one_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data.json"
            data.write_text(
                json.dumps([{**ITEM, "haystack_sessions": [], "answer_session_ids": ["s1"]}])
            )
            old = Path(tmp) / "old.json"
            config = {
                "mode": "none",
                "reader": "gpt-oss:120b",
                "judge": "glm-5.3",
                "endpoint": "https://ollama.com/api/chat",
                "aggregate_limit": None,
                "min_relevance": None,
                "jev_model": None,
                "embedding_model": None,
            }
            row = {
                "question_id": "q1",
                "correct": True,
                "abstention": False,
                "question_type": "multi-session",
            }
            old.write_text(json.dumps({"config": config, "rows": [row]}))
            base = [
                "x",
                "--data",
                str(data),
                "--mode",
                "none",
                "--per-type",
                "0",
                "--resume",
                str(old),
                "--workers",
                "1",
            ]
            with (
                patch.object(qa, "embedding_model", lambda: None),
                patch("sys.stdout"),
                patch("sys.stderr"),
            ):
                with patch.object(sys, "argv", base + ["--samples", "1"]):
                    self.assertEqual(qa.main(), 0)
                with patch.object(sys, "argv", base), self.assertRaises(SystemExit):
                    qa.main()


if __name__ == "__main__":
    unittest.main()
