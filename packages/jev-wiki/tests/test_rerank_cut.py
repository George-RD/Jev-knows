"""The JEV rerank's relevance cut and lexical backfill in Engine.recall."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jev_wiki.engine import Engine
from test_lifecycle import LifecycleDecisionFixture

CLAIMS = {
    "Atlas ships in March.": 3.0,
    "Atlas needs one more reviewer.": 0.4,
    "Atlas docs live in the wiki.": 1.2,
}


class ScriptedRelevance(LifecycleDecisionFixture):
    """Scores each rank question by the claim text it names, as scripted in CLAIMS."""

    def ask(self, state, questions):
        answers = super().ask(state, questions)
        for i, text in state_items(self.calls[-1][0]):
            if f"rank_{i}" in answers:
                answers[f"rank_{i}"]["value"] = CLAIMS[text]
        return answers


def state_items(state: dict):
    return ((int(i), text) for i, text in state.items() if i.isdigit())


class RerankCutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-rerank-")
        self.addCleanup(temporary.cleanup)
        self.engine = Engine(Path(temporary.name), ScriptedRelevance())
        self.engine.ingest("\n\n".join(CLAIMS), source_key="notes")

    def texts(self, **kwargs):
        result = self.engine.recall("Atlas", limit=10, **kwargs)
        self.assertEqual(result["mode"], "jev_reranked")
        return [item["text"] for item in result["items"]]

    def lexical(self):
        return [i["text"] for i in self.engine.recall("Atlas", limit=10, offline=True)["items"]]

    def test_default_cut_keeps_only_claims_scoring_at_least_one_and_a_half(self):
        self.assertEqual(self.texts(), ["Atlas ships in March."])

    def test_lower_cut_keeps_more_in_score_order(self):
        self.assertEqual(
            self.texts(min_relevance=1.0),
            ["Atlas ships in March.", "Atlas docs live in the wiki."],
        )

    def test_zero_cut_reorders_every_shortlisted_claim(self):
        self.assertEqual(
            self.texts(min_relevance=0),
            [
                "Atlas ships in March.",
                "Atlas docs live in the wiki.",
                "Atlas needs one more reviewer.",
            ],
        )

    def test_backfill_keeps_the_rest_in_lexical_order(self):
        promoted = "Atlas ships in March."
        rest = [text for text in self.lexical() if text != promoted]
        self.assertEqual(len(rest), 2)
        self.assertEqual(self.texts(backfill=True), [promoted, *rest])

    def test_backfill_returns_every_lexical_claim(self):
        self.assertCountEqual(self.texts(min_relevance=3, backfill=True), self.lexical())

    def test_invalid_cut_is_rejected(self):
        for value in (-0.1, 3.5, float("nan"), True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.engine.recall("Atlas", min_relevance=value)


if __name__ == "__main__":
    unittest.main()
