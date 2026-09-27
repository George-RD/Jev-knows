"""Independent lifecycle checks with explicitly scripted semantic decisions.

These tests verify state transitions and evidence integrity. The provider below is
not a model and its outcomes supply no evidence of live JEV semantic performance.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jev_wiki.engine import Engine, candidate_spans
from jev_wiki.provider import ProviderError


class LifecycleDecisionFixture:
    """Respond by the engine's question contract, never by simulated NLP."""

    model = "fixture-not-live-jev"

    def __init__(
        self,
        *,
        keep="keep",
        kind="fact",
        confidence=0.99,
        relation="related",
        fail_at=None,
        malformed=None,
        before_answer=None,
    ):
        self.keep = keep
        self.kind = kind
        self.confidence = confidence
        self.relation = relation
        self.fail_at = fail_at
        self.malformed = malformed
        self.before_answer = before_answer
        self.calls = []

    def ask(self, state, questions):
        self.calls.append((json.loads(state), questions))
        if self.before_answer is not None:
            callback, self.before_answer = self.before_answer, None
            callback()
        if len(self.calls) == self.fail_at:
            raise ProviderError("scripted outage; no remote service was called")
        if self.malformed is not None:
            return self.malformed(questions)
        answers = {}
        for name, question in questions.items():
            if name.startswith("keep_"):
                value = self.keep
            elif name.startswith("kind_"):
                value = self.kind
            elif name.startswith("topic_"):
                value = "projects"
            elif name.startswith("rank_"):
                self_assert_score(question)
                value = 3.0
            elif name == "relation":
                value = self.relation
            else:
                raise AssertionError(f"Unrecognized engine question: {name}")
            answers[name] = {"value": value, "confidence": self.confidence}
        return answers


def self_assert_score(question):
    if question.get("type") != "score" or len(question.get("levels", [])) != 4:
        raise AssertionError("Recall fixture needs the expected four-level score")


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-lifecycle-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.provider = LifecycleDecisionFixture()
        self.engine = Engine(self.root, self.provider)

    def ingest(self, text, key="source-one", **kwargs):
        return self.engine.ingest(text, source_key=key, **kwargs)

    def test_citations_resolve_to_exact_unicode_source_spans(self):
        text = "  Café hardware uses Ω connectors.\n\n\tProject Atlas prefers reversible changes.  "
        result = self.ingest(text)
        self.assertEqual(result["status"], "complete")
        recalled = self.engine.recall("Café Atlas", offline=True)
        self.assertEqual(len(recalled["items"]), 2)
        for item in recalled["items"]:
            relative, coordinates = item["citation"].split("#chars=")
            start, end = map(int, coordinates.split("-"))
            raw = (self.root / relative).read_text(encoding="utf-8")
            self.assertEqual(raw, text)
            self.assertEqual(raw[start:end], item["text"])
            self.assertEqual(item["source_id"], result["source_id"])

    def test_capture_and_completed_processing_are_idempotent(self):
        first = self.ingest("Project Atlas has a blue console.")
        first_claim_ids = [c["id"] for c in self.engine.store.claims()]
        calls = len(self.provider.calls)
        again = self.ingest("Project Atlas has a blue console.")
        self.assertEqual(again["source_id"], first["source_id"])
        self.assertEqual(len(self.provider.calls), calls)
        self.assertEqual([c["id"] for c in self.engine.store.claims()], first_claim_ids)

    def test_revision_hides_old_evidence_and_preserves_new_citations(self):
        old = self.ingest("Project Atlas console finish is cobalt.")
        new = self.ingest("Project Atlas console finish is amber.")
        self.assertNotEqual(old["source_id"], new["source_id"])
        items = self.engine.recall("Atlas console", offline=True)["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["source_id"], new["source_id"])
        self.assertNotIn("cobalt", items[0]["text"])

    def test_forgetting_hides_source_without_removing_unrelated_memory(self):
        self.ingest("Project Atlas console finish is cobalt.")
        self.ingest("Project Harbor radio finish is amber.", "source-two")
        self.engine.forget("source-one")
        self.assertEqual(self.engine.recall("Atlas", offline=True)["items"], [])
        self.assertEqual(len(self.engine.recall("Harbor", offline=True)["items"]), 1)

    def test_provider_failure_retains_source_and_defers_processing(self):
        self.engine.provider = LifecycleDecisionFixture(fail_at=1)
        text = "Project Atlas has a blue console."
        result = self.ingest(text)
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(self.engine.store.read_source(result["source_id"]), text)
        self.assertEqual(self.engine.recall("Atlas", offline=True)["items"], [])
        self.engine.provider = self.provider
        retried = self.engine.process(result["source_id"])
        self.assertEqual(retried["status"], "complete")
        self.assertEqual(len(self.engine.recall("Atlas", offline=True)["items"]), 1)

    def test_pending_worker_rotates_past_persistently_deferred_sources(self):
        from jev_wiki.cli import _process_pending

        engine = Engine(self.root / "offline")
        ids = [engine.store.capture(f"Note {i}.", f"note-{i}")["id"] for i in range(3)]
        first = _process_pending(engine, limit=2)
        second = _process_pending(engine, limit=2)
        self.assertEqual(first["deferred"], 2)
        tried_first = {result["source_id"] for result in first["results"]}
        tried_second = {result["source_id"] for result in second["results"]}
        self.assertEqual(len(tried_first), 2)
        self.assertEqual(set(ids) - tried_first, tried_second - tried_first)

    def test_failure_in_later_batch_does_not_publish_partial_claims(self):
        self.engine.provider = LifecycleDecisionFixture(fail_at=2)
        text = "\n\n".join(f"Project Atlas component {number} is reserved." for number in range(5))
        result = self.ingest(text)
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(self.engine.recall("Atlas", offline=True)["items"], [])

    def test_missing_and_malformed_decisions_defer_without_publishing(self):
        fixtures = [
            lambda questions: {},
            lambda questions: None,
            lambda questions: {name: None for name in questions},
            lambda questions: {
                name: {"value": "invented", "confidence": 0.99} for name in questions
            },
        ]
        for index, malformed in enumerate(fixtures):
            with self.subTest(malformed=index):
                engine = Engine(
                    self.root / str(index), LifecycleDecisionFixture(malformed=malformed)
                )
                result = engine.ingest("Project Atlas has a blue console.", "one")
                self.assertEqual(result["status"], "deferred")
                self.assertEqual(engine.recall("Atlas", offline=True)["items"], [])

    def test_no_provider_captures_locally_without_inventing_claims(self):
        engine = Engine(self.root / "offline")
        result = engine.ingest("Project Atlas has a blue console.", "one")
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(result["reason"], "provider_not_configured")
        self.assertEqual(engine.recall("Atlas")["items"], [])

    def test_low_confidence_is_review_only(self):
        self.engine.provider = LifecycleDecisionFixture(confidence=0.40)
        result = self.ingest("Project Atlas has a blue console.")
        self.assertEqual(result["review"], 1)
        self.assertEqual(result["active"], 0)
        self.assertEqual(self.engine.recall("Atlas", offline=True)["items"], [])

    def test_hypothesis_remains_review_even_with_confident_keep(self):
        self.engine.provider = LifecycleDecisionFixture(kind="uncertain")
        result = self.ingest("Project Atlas might use a blue console.")
        self.assertEqual(result["review"], 1)
        self.assertEqual(self.engine.recall("Atlas", offline=True)["items"], [])

    def test_assistant_tool_and_synthesis_are_not_promoted_to_facts(self):
        for role in ("assistant", "tool", "synthesis"):
            with self.subTest(role=role):
                result = self.ingest(
                    f"Project Atlas has a blue console, according to {role}.",
                    key=role,
                    metadata={"role": role},
                )
                self.assertEqual(result["review"], 1)
                self.assertEqual(result["active"], 0)
        self.assertEqual(self.engine.recall("Atlas", offline=True)["items"], [])

    def test_maintained_conflict_is_visible_without_overwriting_evidence(self):
        self.engine.provider = LifecycleDecisionFixture(relation="conflict")
        self.ingest("Project Atlas console finish is cobalt.")
        self.ingest("Project Atlas console finish is amber.", "source-two")
        result = self.engine.maintain()
        self.assertEqual(result["conflicts"], 1)
        recalled = self.engine.recall("Atlas console", offline=True)
        self.assertEqual(len(recalled["items"]), 2)
        self.assertIn("CONFLICT", recalled["context"])
        self.assertTrue(all(item["conflicts"] for item in recalled["items"]))
        self.assertEqual({c["status"] for c in self.engine.store.claims()}, {"active"})

    def test_maintenance_outage_preserves_evidence_and_defers(self):
        self.ingest("Project Atlas console finish is cobalt.")
        self.ingest("Project Atlas console finish is amber.", "source-two")
        self.engine.provider = LifecycleDecisionFixture(fail_at=1)
        result = self.engine.maintain()
        self.assertTrue(result["deferred"])
        self.assertEqual(result["checked_pairs"], 0)
        recalled = self.engine.recall("Atlas console", offline=True)
        self.assertEqual(len(recalled["items"]), 2)
        self.assertTrue(all(not item["conflicts"] for item in recalled["items"]))

    def test_uncertain_relationship_does_not_become_a_conflict(self):
        self.ingest("Project Atlas console finish is cobalt.")
        self.ingest("Project Atlas console finish is amber.", "source-two")
        self.engine.provider = LifecycleDecisionFixture(relation="conflict", confidence=0.4)
        result = self.engine.maintain()
        self.assertEqual(result["conflicts"], 0)
        relations = [r for claim in self.engine.store.claims() for r in claim["relations"]]
        self.assertTrue(relations)
        self.assertTrue(all(relation["type"] == "uncertain" for relation in relations))

    def test_reprocessing_complete_source_preserves_conflict_links(self):
        self.engine.provider = LifecycleDecisionFixture(relation="conflict")
        first = self.ingest("Project Atlas console finish is cobalt.")
        self.ingest("Project Atlas console finish is amber.", "source-two")
        self.engine.maintain()
        self.engine.process(first["source_id"])
        recalled = self.engine.recall("Atlas console", offline=True)
        self.assertTrue(all(item["conflicts"] for item in recalled["items"]))

    def test_forgetting_conflict_target_removes_dangling_warnings(self):
        self.engine.provider = LifecycleDecisionFixture(relation="conflict")
        self.ingest("Project Atlas console finish is cobalt.")
        self.ingest("Project Atlas console finish is amber.", "source-two")
        self.engine.maintain()
        self.engine.forget("source-two")
        recalled = self.engine.recall("Atlas console", offline=True)
        self.assertEqual(len(recalled["items"]), 1)
        self.assertEqual(recalled["items"][0]["conflicts"], [])
        self.assertNotIn("CONFLICT", recalled["context"])

    def test_revising_conflict_target_removes_old_revision_warnings(self):
        self.engine.provider = LifecycleDecisionFixture(relation="conflict")
        self.ingest("Project Atlas console finish is cobalt.")
        self.ingest("Project Atlas console finish is amber.", "source-two")
        self.engine.maintain()
        self.ingest("Project Harbor radio finish is amber.", "source-two")
        recalled = self.engine.recall("Atlas console", offline=True)
        self.assertEqual(len(recalled["items"]), 1)
        self.assertEqual(recalled["items"][0]["conflicts"], [])

    def test_forget_during_rerank_cannot_return_forgotten_evidence(self):
        self.ingest("Project Atlas console finish is cobalt.")
        self.engine.provider = LifecycleDecisionFixture(
            before_answer=lambda: self.engine.forget("source-one")
        )
        self.assertEqual(self.engine.recall("Atlas console")["items"], [])

    def test_revision_during_rerank_cannot_return_replaced_evidence(self):
        self.ingest("Project Atlas console finish is cobalt.")
        self.engine.provider = LifecycleDecisionFixture(
            before_answer=lambda: self.engine.store.capture(
                "Project Atlas console finish is amber.", "source-one"
            )
        )
        self.assertEqual(self.engine.recall("Atlas console")["items"], [])

    def test_forget_during_processing_cannot_resurrect_source(self):
        self.engine.provider = LifecycleDecisionFixture(
            before_answer=lambda: self.engine.forget("source-one")
        )
        result = self.ingest("Project Atlas console finish is cobalt.")
        self.assertNotEqual(result["status"], "complete")
        self.assertEqual(self.engine.recall("Atlas console", offline=True)["items"], [])

    def test_render_reconstruction_is_deterministic_without_provider(self):
        self.ingest("Project Atlas console finish is cobalt.")
        pages = {
            path.relative_to(self.root): path.read_bytes()
            for path in self.root.glob("wiki/**/*.md")
        }
        self.assertTrue(pages)
        for relative in pages:
            (self.root / relative).unlink()
        Engine(self.root).store.render()
        rebuilt = {
            path.relative_to(self.root): path.read_bytes()
            for path in self.root.glob("wiki/**/*.md")
        }
        self.assertEqual(rebuilt, pages)

    def test_recall_is_bounded_and_never_truncates_a_quotation(self):
        text = "\n\n".join(f"Atlas {number}: " + "complete evidence " * 30 for number in range(5))
        self.ingest(text)
        small = self.engine.recall("Atlas", max_chars=256, offline=True)
        self.assertEqual(small["items"], [])
        self.assertEqual(small["context"], "")
        result = self.engine.recall("Atlas", limit=2, max_chars=1800, offline=True)
        self.assertLessEqual(len(result["items"]), 2)
        self.assertLessEqual(len(result["context"]), 1800)
        self.assertGreater(len(result["items"]), 0)
        candidates = {span["text"] for span in candidate_spans(text)}
        self.assertTrue(all(item["text"] in candidates for item in result["items"]))

    def test_unmatched_query_does_not_get_unrelated_padding(self):
        self.ingest("Project Atlas has a blue console.")
        result = self.engine.recall("ornithology")
        self.assertEqual(result["items"], [])
        self.assertEqual(result["candidate_count"], 0)

    def test_rerank_failure_falls_back_to_exact_lexical_evidence(self):
        self.ingest("Project Atlas has a blue console.")
        self.engine.provider = LifecycleDecisionFixture(fail_at=1)
        result = self.engine.recall("Atlas")
        self.assertTrue(result["degraded"])
        self.assertEqual(result["mode"], "lexical")
        self.assertEqual(len(result["items"]), 1)

    def test_live_adapter_is_not_claimed_for_fixture_decisions(self):
        self.ingest("Project Atlas has a blue console.")
        claim = self.engine.store.claims()[0]
        self.assertEqual(claim["provider"], "LifecycleDecisionFixture")
        self.assertEqual(claim["model"], "fixture-not-live-jev")


if __name__ == "__main__":
    unittest.main()
