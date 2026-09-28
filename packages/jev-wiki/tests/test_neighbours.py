"""Neighbour recall: the claims next to a recalled claim come with it."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jev_wiki.engine import MIN_NEIGHBOURS, Engine
from test_aggregate import Scored
from test_lifecycle import LifecycleDecisionFixture

PUPPY = (
    "I'm looking for some advice on my garden this spring.\n\n"
    "I'm thinking of getting more supplies for my puppy, Luna. "
    "She's still in potty-training, and I've been using eco-friendly training pads. "
    "I got a set of 10 for $25 about a month ago, and they've been a lifesaver.\n\n"
    "Do you have any tips for growing tomatoes in pots?"
)
PADS = "She's still in potty-training, and I've been using eco-friendly training pads."
PRICE = "I got a set of 10 for $25 about a month ago, and they've been a lifesaver."
LUNA = "I'm thinking of getting more supplies for my puppy, Luna."


class NeighbourRecallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-neighbours-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engine = Engine(self.root, LifecycleDecisionFixture())
        self.engine.ingest(PUPPY, source_key="chat")

    def test_off_by_default(self):
        result = self.engine.recall("training pads", limit=1, offline=True)
        self.assertEqual([i["text"] for i in result["items"]], [PADS])
        self.assertEqual(result["neighbours"], 0)

    def test_the_next_claim_then_the_previous_follow_their_anchor(self):
        result = self.engine.recall("training pads", limit=1, offline=True, neighbours=True)
        self.assertEqual([i["text"] for i in result["items"]], [PADS, PRICE, LUNA])
        anchor = result["items"][0]["id"]
        self.assertNotIn("neighbour_of", result["items"][0])
        self.assertEqual([i.get("neighbour_of") for i in result["items"][1:]], [anchor] * 2)
        self.assertEqual(result["neighbours"], 2)
        self.assertEqual(result["context"].count("\n["), 3)

    def test_neighbours_do_not_use_up_the_limit(self):
        runs = [f"Run {i}: I did a charity fun run this spring." for i in range(12)]
        engine = Engine(self.root, LifecycleDecisionFixture())
        engine.ingest("\n\n".join(runs), source_key="runs")
        query = "charity fun run"
        plain = engine.recall(query, limit=4, max_chars=20_000, offline=True)
        near = engine.recall(query, limit=4, max_chars=20_000, offline=True, neighbours=True)
        ranked = [i["id"] for i in near["items"] if "neighbour_of" not in i]
        # A ranked claim already added as a neighbour is not repeated, so the next
        # ranked claim takes its place.
        self.assertEqual(len(ranked), 4)
        self.assertEqual(ranked[0], plain["items"][0]["id"])
        self.assertEqual(near["neighbours"], MIN_NEIGHBOURS)  # max(5, 4 // 2)
        self.assertEqual(len(near["items"]), 4 + MIN_NEIGHBOURS)
        self.assertEqual(len({i["id"] for i in near["items"]}), len(near["items"]))

    def test_neighbours_use_at_most_a_quarter_of_the_budget(self):
        result = self.engine.recall(
            "training pads", limit=1, max_chars=600, offline=True, neighbours=True
        )
        # The header and the anchor fit in 600; one 150-char neighbour block does not.
        self.assertEqual([i["text"] for i in result["items"]], [PADS])
        roomy = self.engine.recall(
            "training pads", limit=1, max_chars=1_200, offline=True, neighbours=True
        )
        self.assertEqual([i["text"] for i in roomy["items"]], [PADS, PRICE])
        self.assertLessEqual(len(roomy["context"]), 1_200)

    def test_other_sources_are_not_neighbours(self):
        self.engine.ingest("I also bought training pads for the neighbour's dog.", "other")
        result = self.engine.recall("training pads", limit=2, offline=True, neighbours=True)
        by_id = {i["id"]: i for i in result["items"]}
        for item in result["items"]:
            if "neighbour_of" in item:
                self.assertEqual(item["source_id"], by_id[item["neighbour_of"]]["source_id"])

    def test_claims_the_ranker_rejected_never_return_as_neighbours(self):
        runs = [f"Run {i}: I did a charity fun run this spring." for i in range(40)]
        engine = Engine(self.root, LifecycleDecisionFixture())
        engine.ingest("\n\n".join(runs), source_key="runs")
        provider = Scored()
        engine = Engine(self.root, provider)
        result = engine.recall("charity fun run", limit=12, max_chars=20_000, neighbours=True)
        ranked_state, _ = provider.calls[-1]
        rejected = {text for i, text in ranked_state.items() if int(i) % 2 == 0}
        self.assertTrue(rejected)
        self.assertFalse(rejected & {i["text"] for i in result["items"]})
        self.assertGreater(result["neighbours"], 0)  # Claims the ranker never saw.


class CliNeighbourTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-neighbours-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        Engine(self.root, LifecycleDecisionFixture()).ingest(PUPPY, source_key="chat")

    def recall(self, *extra):
        from jev_wiki.cli import _parser, run

        base = ["--root", str(self.root), "--provider", "none", "recall", "training pads"]
        return run(_parser().parse_args([*base, "--offline", "--limit", "1", *extra]))

    def test_cli_recall_adds_neighbours_by_default(self):
        self.assertEqual([i["text"] for i in self.recall()["items"]], [PADS, PRICE, LUNA])

    def test_no_neighbours_turns_it_off(self):
        self.assertEqual([i["text"] for i in self.recall("--no-neighbours")["items"]], [PADS])


if __name__ == "__main__":
    unittest.main()
