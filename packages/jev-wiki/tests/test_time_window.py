"""Dated recall: sources carry a date, and "N weeks ago" packs that window's claims first."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from jev_wiki.engine import Engine, source_date, time_window
from test_lifecycle import LifecycleDecisionFixture

TUESDAY = date(2023, 5, 30)


def span(query: str, as_of: date = TUESDAY) -> tuple[str, str] | None:
    window = time_window(query, as_of)
    return (window["start"], window["end"]) if window else None


class TimeWindowTests(unittest.TestCase):
    def test_counted_units_ago_allow_rounding(self):
        self.assertEqual(span("What did I buy 10 days ago?"), ("2023-05-19", "2023-05-21"))
        self.assertEqual(span("What did I say two weeks ago?"), ("2023-05-13", "2023-05-19"))
        self.assertEqual(span("Where was I a month ago?"), ("2023-04-20", "2023-05-10"))
        self.assertEqual(span("a couple of days ago"), ("2023-05-27", "2023-05-29"))

    def test_named_days_resolve_to_the_most_recent_one(self):
        self.assertEqual(span("Who called yesterday?"), ("2023-05-29", "2023-05-29"))
        self.assertEqual(span("lunch last Tuesday"), ("2023-05-22", "2023-05-24"))
        self.assertEqual(span("dinner last Monday"), ("2023-05-28", "2023-05-30"))
        self.assertEqual(span("What did I cook last weekend?"), ("2023-05-26", "2023-05-29"))

    def test_last_period_is_the_calendar_one_and_the_past_period_runs_to_today(self):
        self.assertEqual(span("What did I finish last month?"), ("2023-04-01", "2023-04-30"))
        self.assertEqual(span("Where did I go last week?"), ("2023-05-22", "2023-05-28"))
        self.assertEqual(span("last year", date(2023, 1, 5)), ("2022-01-01", "2022-12-31"))
        self.assertEqual(span("How many in the past month?"), ("2023-04-30", "2023-05-30"))
        self.assertEqual(span("in the last week"), ("2023-05-23", "2023-05-30"))

    def test_undated_questions_have_no_window(self):
        for query in ("What is my dog's name?", "Which did I start first?", "the last time"):
            self.assertIsNone(span(query), query)

    def test_source_date_prefers_the_callers_date(self):
        self.assertEqual(
            source_date({"metadata": {"date": "2023-03-04"}, "created_at": "2026-09-28T01:00"}),
            "2023-03-04",
        )
        self.assertEqual(
            source_date({"metadata": {"date": "not a date"}, "created_at": "2026-09-28T01:00"}),
            "2026-09-28",
        )
        self.assertIsNone(source_date({"metadata": {}}))


class DatedRecallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-dated-")
        self.addCleanup(temporary.cleanup)
        self.engine = Engine(Path(temporary.name), LifecycleDecisionFixture())
        # The run in the named window is the least lexically similar of the three.
        for key, day, text in (
            ("a", "2023-05-01", "I did a charity fun run in town, a charity fun run."),
            ("b", "2023-05-16", "I did a run."),
            ("c", "2023-05-25", "I did a charity fun run with my sister."),
        ):
            self.engine.ingest(text, source_key=key, metadata={"date": day})

    def test_items_and_context_carry_the_source_date(self):
        result = self.engine.recall("charity fun run", offline=True)
        self.assertEqual(
            {i["date"] for i in result["items"]}, {"2023-05-01", "2023-05-16", "2023-05-25"}
        )
        self.assertIn("(fact, 2023-05-01)", result["context"])
        self.assertIsNone(result["time_window"])

    def test_claims_from_the_named_window_come_first(self):
        query = "Which charity fun run did I do two weeks ago?"
        plain = self.engine.recall(query, offline=True, as_of="2023-01-01")
        dated = self.engine.recall(query, offline=True, as_of=TUESDAY)
        self.assertEqual(plain["items"][0]["date"], "2023-05-01")
        self.assertEqual(dated["items"][0]["date"], "2023-05-16")
        self.assertEqual(dated["time_window"]["phrase"], "two weeks ago")
        # Reordered only: the same claims come back.
        self.assertEqual(
            sorted(i["id"] for i in plain["items"]), sorted(i["id"] for i in dated["items"])
        )

    def test_invalid_as_of_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.recall("runs two weeks ago", offline=True, as_of="soon")


if __name__ == "__main__":
    unittest.main()
