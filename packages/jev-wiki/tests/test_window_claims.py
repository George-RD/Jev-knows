"""Window claims: a question naming a date gets the claims its window's sources hold."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jev_wiki.engine import MAX_WINDOW_CLAIMS, Engine
from test_lifecycle import LifecycleDecisionFixture

QUERY = "What kitchen appliance did I buy 10 days ago?"
AS_OF = "2023-03-25"  # Ten days after the smoker.
SMOKER = "By the way, I just got a smoker today and I'm excited to try it."


class WindowClaimTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-window-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engine = Engine(self.root, LifecycleDecisionFixture())
        self.engine.ingest(
            f"I'm looking for some new BBQ sauce recipes to try out.\n\n{SMOKER}",
            source_key="bbq",
            metadata={"date": "2023-03-15"},
        )
        self.engine.ingest(
            "I bought a new kitchen appliance for the office last year.",
            source_key="office",
            metadata={"date": "2022-06-01"},
        )

    def recall(self, **options):
        return self.engine.recall(QUERY, offline=True, as_of=AS_OF, **options)

    def test_off_by_default(self):
        result = self.recall()
        self.assertNotIn(SMOKER, [i["text"] for i in result["items"]])
        self.assertEqual(result["window_claims"], 0)

    def test_the_window_sources_claims_come_back_marked(self):
        result = self.recall(window_claims=True)
        smoker = [i for i in result["items"] if i["text"] == SMOKER]
        self.assertEqual(len(smoker), 1)
        self.assertTrue(smoker[0]["in_window"])
        self.assertEqual(smoker[0]["date"], "2023-03-15")
        self.assertEqual(result["window_claims"], 2)  # Both BBQ-chat claims.
        # The ranked match from another date is still there: extras don't use the limit.
        self.assertIn(
            "I bought a new kitchen appliance for the office last year.",
            [i["text"] for i in result["items"] if "in_window" not in i],
        )

    def test_ranked_claims_keep_their_room(self):
        appliances = [
            f"Kitchen appliance {i}: I bought a blender for the office." for i in range(8)
        ]
        self.engine.ingest("\n\n".join(appliances), "shop", metadata={"date": "2022-01-01"})
        plain = self.recall(limit=5, max_chars=1_000)
        dated = self.recall(limit=5, max_chars=1_000, window_claims=True)
        ranked = [i["id"] for i in dated["items"] if "in_window" not in i]
        self.assertEqual(ranked, [i["id"] for i in plain["items"]])

    def test_nothing_is_added_without_a_named_date(self):
        result = self.engine.recall(
            "What kitchen appliance did I buy?", offline=True, as_of=AS_OF, window_claims=True
        )
        self.assertIsNone(result["time_window"])
        self.assertEqual(result["window_claims"], 0)

    def test_at_most_max_window_claims_within_a_quarter_of_the_budget(self):
        notes = [f"Note {i}: I tidied the garage shelves this morning." for i in range(30)]
        self.engine.ingest("\n\n".join(notes), "garage", metadata={"date": "2023-03-15"})
        result = self.recall(window_claims=True, max_chars=20_000)
        self.assertEqual(result["window_claims"], MAX_WINDOW_CLAIMS)
        small = self.recall(window_claims=True, max_chars=1_000)
        added = sum(
            len(i["text"]) + 60 for i in small["items"] if i.get("in_window")
        )  # Roughly each block's size.
        self.assertLess(small["window_claims"], MAX_WINDOW_CLAIMS)
        self.assertLessEqual(added, 1_000 // 4 + 60)
        self.assertLessEqual(len(small["context"]), 1_000)


class CliWindowClaimTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-window-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        Engine(self.root, LifecycleDecisionFixture()).ingest(
            SMOKER, source_key="bbq", metadata={"date": "2023-03-15"}
        )

    def test_cli_recall_adds_window_claims_by_default(self):
        from jev_wiki.cli import _parser

        base = ["--root", str(self.root), "--provider", "none", "recall", "q"]
        self.assertTrue(_parser().parse_args(base).window_claims)
        self.assertFalse(_parser().parse_args([*base, "--no-window-claims"]).window_claims)


if __name__ == "__main__":
    unittest.main()
