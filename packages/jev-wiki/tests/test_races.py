"""Controlled interleavings of two workers, with no network or timing assumptions."""

import tempfile
import unittest
from pathlib import Path

from jev_wiki.engine import Engine
from jev_wiki.provider import ProviderError


class _DecisionFixture:
    model = "fixture-not-live-jev"

    def __init__(self, before_answer=None, fail=False):
        self.before_answer = before_answer
        self.fail = fail

    def ask(self, state, questions):
        if self.before_answer is not None:
            callback, self.before_answer = self.before_answer, None
            callback()
        if self.fail:
            raise ProviderError("Explicit test outage")
        values = {"keep": "keep", "kind": "fact", "topic": "projects"}
        return {
            name: {"value": values[name.split("_")[0]], "confidence": 0.99} for name in questions
        }


class ProcessingRaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-races-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.text = "Project Atlas uses blue widgets."
        self.worker_b = Engine(self.root, _DecisionFixture())

    def test_late_success_does_not_undo_review_after_other_worker_completed(self):
        def finish_and_review():
            self.worker_b.ingest(self.text, "document")
            claim = self.worker_b.store.claims()[0]
            self.worker_b.store.update_claim(
                claim["id"], {"status": "review", "review_reason": "Human rejected this claim"}
            )

        worker_a = Engine(self.root, _DecisionFixture(before_answer=finish_and_review))
        worker_a.ingest(self.text, "document")

        claims = worker_a.store.claims(active_only=False)
        self.assertEqual(claims[0]["status"], "review")
        self.assertEqual(claims[0]["review_reason"], "Human rejected this claim")
        self.assertEqual(worker_a.recall("Atlas", offline=True)["items"], [])
        self.assertEqual(worker_a.store.sources()[0]["processing"], "complete")

    def test_late_failure_does_not_defer_other_workers_completed_source(self):
        worker_a = Engine(
            self.root,
            _DecisionFixture(
                before_answer=lambda: self.worker_b.ingest(self.text, "document"), fail=True
            ),
        )
        worker_a.ingest(self.text, "document")

        source = worker_a.store.sources()[0]
        self.assertEqual(source["processing"], "complete")
        self.assertNotIn("processing_error", source)
        self.assertEqual(len(worker_a.recall("Atlas", offline=True)["items"]), 1)


if __name__ == "__main__":
    unittest.main()
