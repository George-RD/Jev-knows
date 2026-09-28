"""Aggregation-aware recall: counting and date questions may take more claims."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jev_wiki.engine import MAX_AGGREGATE_LIMIT, Engine, aggregation_query
from test_lifecycle import LifecycleDecisionFixture


class AggregationQueryTests(unittest.TestCase):
    def test_counting_totals_and_dates_are_aggregation(self):
        for query in (
            "How many weddings have I attended this year?",
            "What is the total cost of the car cover and detailing spray?",
            "Which project did I start first, the Ferrari or the Zero?",
            "How many days passed between the MoMA visit and the exhibit?",
            "Which book did I finish a week ago?",
            "How often do I go to the gym?",
        ):
            self.assertTrue(aggregation_query(query), query)

    def test_single_facts_and_requests_are_not(self):
        for query in (
            "What type of cocktail recipe did I try last weekend?",
            "Where did I attend my study abroad program?",
            "Any documentary recommendations for tonight?",
            "Who is the firstborn in the family?",  # "first" only as a whole word
        ):
            self.assertFalse(aggregation_query(query), query)


class Scored(LifecycleDecisionFixture):
    """Ranks every odd candidate as direct evidence and every even one as unrelated."""

    def ask(self, state, questions):
        answers = super().ask(state, questions)
        for name in answers:
            if name.startswith("rank_"):
                answers[name]["value"] = 3.0 if int(name.split("_")[1]) % 2 else 0.0
        return answers


class AggregateRecallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-aggregate-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engine = Engine(self.root, LifecycleDecisionFixture())
        runs = [f"I ran a charity fun run in town number {i} this spring." for i in range(40)]
        self.engine.ingest("\n\n".join(runs), source_key="runs")

    def test_off_by_default(self):
        result = self.engine.recall("How many fun runs did I do?", limit=20, offline=True)
        self.assertEqual(len(result["items"]), 20)
        self.assertFalse(result["aggregate"])

    def test_counting_question_takes_the_aggregate_limit(self):
        result = self.engine.recall(
            "How many fun runs did I do?",
            limit=20,
            max_chars=20_000,
            offline=True,
            aggregate_limit=35,
        )
        self.assertTrue(result["aggregate"])
        self.assertEqual(len(result["items"]), 35)
        self.assertEqual(len({i["id"] for i in result["items"]}), 35)
        self.assertEqual(result["context"].count("\n["), 35)

    def test_other_questions_keep_the_plain_limit(self):
        result = self.engine.recall(
            "Tell me about my charity fun run", limit=5, offline=True, aggregate_limit=35
        )
        self.assertFalse(result["aggregate"])
        self.assertEqual(len(result["items"]), 5)

    def test_context_budget_still_bounds_the_output(self):
        result = self.engine.recall(
            "How many fun runs did I do?",
            limit=20,
            max_chars=1_000,
            offline=True,
            aggregate_limit=MAX_AGGREGATE_LIMIT,
        )
        self.assertLessEqual(len(result["context"]), 1_000)
        self.assertLess(len(result["items"]), 20)

    def test_ranker_request_is_unchanged_and_its_rejects_stay_out(self):
        provider = Scored()
        engine = Engine(self.root, provider)
        result = engine.recall(
            "How many fun runs did I do?", limit=20, max_chars=20_000, aggregate_limit=40
        )
        ranked_state, _ = provider.calls[-1]
        self.assertEqual(len(ranked_state), 24)  # The shortlist, as without aggregation.
        rejected = {text for i, text in ranked_state.items() if int(i) % 2 == 0}
        texts = [i["text"] for i in result["items"]]
        self.assertFalse(rejected & set(texts))
        # 12 ranked keepers first, then the 16 claims the ranker never saw.
        self.assertEqual(len(texts), 12 + 16)
        self.assertTrue(all(i["relevance"] == 3.0 for i in result["items"][:12]))
        self.assertTrue(all(i["relevance"] is None for i in result["items"][12:]))
        self.assertEqual(result["mode"], "jev_reranked")

    def test_invalid_aggregate_limits_are_rejected(self):
        for value in (0, MAX_AGGREGATE_LIMIT + 1, True, 2.5):
            with self.assertRaises(ValueError):
                self.engine.recall("How many runs?", offline=True, aggregate_limit=value)


if __name__ == "__main__":
    unittest.main()
