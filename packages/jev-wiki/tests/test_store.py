"""Storage invariants exercised without a model, network, or database service."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jev_wiki.store import WikiStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "memory"
        self.store = WikiStore(self.root)

    def capture_claim(self, text="George prefers concise answers.", key="profile", **updates):
        source = self.store.capture(text, key)
        claim = {
            "id": f"{source['id']}:0",
            "source_id": source["id"],
            "text": text,
            "start": 0,
            "end": len(text),
            "kind": "preference",
            "topic": "George",
            "status": "active",
            "confidence": 0.9,
            "relations": [],
        }
        claim.update(updates)
        self.store.put_claims(source["id"], [claim])
        return source, claim

    def test_same_key_and_content_is_idempotent_and_keys_are_isolated(self):
        first = self.store.capture("Exact source.\r\nUnicode: café 🧵", "one", metadata={"x": [1]})
        again = self.store.capture("Exact source.\r\nUnicode: café 🧵", "one")
        second = self.store.capture("Exact source.\r\nUnicode: café 🧵", "two")
        self.assertEqual(first, again)
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(first["hash"], second["hash"])
        self.assertEqual(len(self.store.events()), 2)
        self.assertEqual(self.store.read_source(first["id"]), "Exact source.\r\nUnicode: café 🧵")
        first["metadata"]["x"].append(2)
        self.assertEqual(self.store.sources()[0]["metadata"], {"x": [1]})

    def test_revision_invalidates_claims_and_survives_restart(self):
        old, claim = self.capture_claim()
        self.store.render()
        new = self.store.capture("George now prefers detailed answers.", "profile")
        self.assertEqual(self.store.claims(), [])
        self.assertEqual(self.store.claims(False)[0]["review_reason"], "source_superseded")
        self.assertEqual(self.store.sources(), [new])
        self.assertFalse(self.store.is_current(old["id"]))
        self.assertEqual(self.store.active_evidence([claim["id"]]), {})
        reopened = WikiStore(self.root)
        self.assertEqual(len(reopened.sources(False)), 2)
        self.assertEqual(reopened.read_source(old["id"]), claim["text"])
        for page in (self.root / "wiki/generated").glob("*.md"):
            self.assertNotIn(claim["text"], page.read_text())

    def test_reverting_content_requires_fresh_compile(self):
        original, _ = self.capture_claim("First preference.")
        self.store.capture("Changed preference.", "profile")
        returned = self.store.capture("First preference.", "profile")
        self.assertEqual(returned["id"], original["id"])
        self.assertEqual(returned["processing"], "pending")
        self.assertEqual(self.store.claims(), [])

    def test_bad_citation_does_not_partially_replace_claim_batch(self):
        source, original = self.capture_claim()
        valid = {**original, "id": "replacement"}
        bad = {**original, "id": "fabricated", "text": "George likes verbose answers."}
        before = self.store.events()
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.store.put_claims(source["id"], [valid, bad])
        self.assertEqual(self.store.claims(), [original])
        self.assertEqual(self.store.events(), before)
        for start, end in ((-1, 3), (0, 1000), (True, 2), (2, 2)):
            with self.assertRaises(ValueError):
                self.store.put_claims(source["id"], [{**original, "start": start, "end": end}])

    def test_claim_ids_cannot_overwrite_another_sources_evidence(self):
        _, original = self.capture_claim()
        other = self.store.capture(original["text"], "another")
        with self.assertRaisesRegex(ValueError, "another source"):
            self.store.put_claims(other["id"], [{**original, "source_id": other["id"]}])
        self.assertEqual(self.store.claims(), [original])

    def test_review_claims_and_review_updates(self):
        source, claim = self.capture_claim(
            status="review", decisions={"sensitive": True}, provider="fixture"
        )
        self.assertEqual(self.store.claims(), [])
        self.assertEqual(self.store.claims(False)[0]["decisions"], {"sensitive": True})
        self.store.update_claim(
            claim["id"],
            {
                "status": "active",
                "relations": [{"type": "conflicts_with", "target": "another-claim"}],
            },
        )
        self.assertEqual(len(self.store.claims()), 1)
        with self.assertRaises(ValueError):
            self.store.update_claim(claim["id"], {"text": "rewritten"})
        with self.assertRaises(ValueError):
            self.store.update_claim(claim["id"], {"source_id": "another"})
        self.assertIn(claim["id"], self.store.active_evidence([claim["id"], "missing"]))
        self.store.set_processing(source["id"], "deferred", "Provider unavailable")
        self.assertEqual(self.store.sources()[0]["processing_error"], "Provider unavailable")
        self.store.set_processing(source["id"], "complete")
        self.assertNotIn("processing_error", self.store.sources()[0])

    def test_repeated_deferral_does_not_grow_the_event_log(self):
        source = self.store.capture("Deferred text.", "deferred")
        self.store.set_processing(source["id"], "deferred", "provider_not_configured")
        state_path = self.root / "state.json"
        events = len(json.loads(state_path.read_text())["events"])
        for _ in range(5):
            self.store.set_processing(source["id"], "deferred", "provider_not_configured")
        self.assertEqual(len(json.loads(state_path.read_text())["events"]), events)
        self.assertIn("processing_attempted_at", self.store.sources()[0])

    def test_forget_tombstones_all_revisions_and_regenerates_pages(self):
        old, _ = self.capture_claim("A forgotten preference.")
        newer = self.store.capture("A newer forgotten preference.", "profile")
        self.store.put_claims(
            newer["id"],
            [{"id": "latest", "text": "A newer forgotten preference.", "start": 0, "end": 29}],
        )
        self.store.render()
        tombstone = self.store.forget("profile")
        self.assertEqual(set(tombstone["source_ids"]), {old["id"], newer["id"]})
        self.assertEqual(self.store.sources(), [])
        self.assertEqual(self.store.claims(), [])
        self.assertEqual(self.store.active_evidence(["latest"]), {})
        self.assertEqual(self.store.read_source(old["id"]), "A forgotten preference.")
        for page in (self.root / "wiki/generated").glob("*.md"):
            self.assertNotIn("forgotten preference", page.read_text())
        with self.assertRaisesRegex(ValueError, "forgotten"):
            self.store.capture("Even brand new content.", "profile")
        with self.assertRaisesRegex(ValueError, "inactive"):
            self.store.put_claims(newer["id"], [])
        events = self.store.events()
        self.assertEqual(self.store.forget("profile"), tombstone)
        self.assertEqual(self.store.events(), events)

    def test_render_preserves_handwritten_pages_and_handwritten_index(self):
        self.capture_claim(topic="../../escape [topic]")
        wiki = self.root / "wiki"
        wiki.mkdir()
        (wiki / "notes.md").write_text("My notes, never overwrite.")
        (wiki / "index.md").write_text("My handwritten index.")
        result = self.store.render()
        self.assertIn("wiki/index.md", result["preserved"])
        self.assertEqual((wiki / "index.md").read_text(), "My handwritten index.")
        self.assertEqual((wiki / "notes.md").read_text(), "My notes, never overwrite.")
        pages = list((wiki / "generated").glob("escape-topic-*.md"))
        self.assertEqual(len(pages), 1)
        self.assertIn("George prefers concise answers.", pages[0].read_text())
        self.assertIn("../../raw/", pages[0].read_text())
        self.assertTrue((wiki / "generated/index.md").exists())
        self.assertEqual(self.store.lint(), [])

    def test_hash_tampering_blocks_read_recall_and_render(self):
        source, claim = self.capture_claim()
        (self.root / source["path"]).write_text("Tampered content.")
        for action in (
            lambda: self.store.read_source(source["id"]),
            self.store.claims,
            lambda: self.store.active_evidence([claim["id"]]),
            self.store.render,
        ):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                action()
        self.assertIn("invalid_source", {issue["code"] for issue in self.store.lint()})

    def test_symlink_raw_directory_is_rejected_without_external_write(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (self.root / "raw").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.store.capture("No escape.", "../../escape")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(self.store.sources(), [])

    def test_symlink_raw_file_is_rejected(self):
        source, _ = self.capture_claim()
        raw = self.root / source["path"]
        external = Path(self.temporary.name) / "external"
        external.write_bytes(raw.read_bytes())
        raw.unlink()
        raw.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.store.read_source(source["id"])

    def test_empty_successful_compile_has_explicit_processing_status(self):
        source = self.store.capture("Irrelevant chatter.", "empty")
        self.store.put_claims(source["id"], [])
        self.store.set_processing(source["id"], "complete")
        self.assertEqual(WikiStore(self.root).sources()[0]["processing"], "complete")
        self.assertEqual(self.store.claims(), [])

    def test_concurrent_processes_do_not_lose_captures_or_events(self):
        script = """import sys
from jev_wiki.store import WikiStore
store = WikiStore(sys.argv[1])
for index in range(8):
    key = f'{sys.argv[2]}:{index}'
    store.capture(f'Evidence {key}', key)
"""
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        workers = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(self.root), str(worker)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            for worker in range(4)
        ]
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=30)
            self.assertEqual(worker.returncode, 0, (stdout + stderr).decode())
        self.assertEqual(len(self.store.sources()), 32)
        self.assertEqual([event["id"] for event in self.store.events()], list(range(1, 33)))
        self.assertEqual(self.store.lint(), [])

    def test_state_limit_is_checked_before_parsing(self):
        with (self.root / "state.json").open("wb") as handle:
            handle.truncate(16 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(ValueError, "16 MiB"):
            self.store.sources()

    def test_tampered_claim_fails_final_evidence_validation(self):
        _, claim = self.capture_claim()
        state_path = self.root / "state.json"
        state = json.loads(state_path.read_text())
        state["claims"][claim["id"]]["text"] = "Invented claim."
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.store.active_evidence([claim["id"]])

    def test_relations_are_symmetric_and_hidden_after_target_revision(self):
        _, left = self.capture_claim("Atlas has a cobalt console.", "left")
        _, right = self.capture_claim("Atlas has an amber console.", "right")
        self.assertTrue(
            self.store.relate(left["id"], right["id"], {"type": "conflict", "confidence": 0.99})
        )
        active = self.store.active_evidence([left["id"], right["id"]])
        self.assertEqual(active[left["id"]]["relations"][0]["target"], right["id"])
        self.assertEqual(active[right["id"]]["relations"][0]["target"], left["id"])
        self.store.capture("Atlas has a red console.", "right")
        self.assertEqual(self.store.claims()[0]["relations"], [])
        self.assertEqual(self.store.active_evidence([left["id"]])[left["id"]]["relations"], [])
        before = self.store.events()
        self.assertFalse(self.store.relate(left["id"], right["id"], {"type": "conflict"}))
        self.assertEqual(self.store.events(), before)
        self.assertTrue(
            next(claim for claim in self.store.claims(False) if claim["id"] == left["id"])[
                "relations"
            ]
        )

    def test_lint_detects_projection_body_tampering_and_missing_pages(self):
        self.capture_claim()
        self.store.render()
        topic = next((self.root / "wiki/generated").glob("george-*.md"))
        topic.write_text(topic.read_text() + "\nAn invented addition.\n")
        (self.root / "wiki/generated/review.md").unlink()
        issues = self.store.lint()
        self.assertIn("stale_projection", {issue["code"] for issue in issues})
        self.assertIn("missing_projection", {issue["code"] for issue in issues})
        self.store.render()
        self.assertEqual(self.store.lint(), [])

    def test_non_regular_raw_source_is_rejected_without_blocking(self):
        source, _ = self.capture_claim()
        raw = self.root / source["path"]
        raw.unlink()
        os.mkfifo(raw)
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.store.read_source(source["id"])

    def test_late_compile_cannot_override_completed_human_review(self):
        source = self.store.capture("Atlas console is cobalt.", "atlas")
        candidate = {"id": "atlas-claim", "text": "Atlas console is cobalt.", "start": 0, "end": 24}
        self.assertTrue(self.store.complete_processing(source["id"], [candidate]))
        self.store.update_claim(
            "atlas-claim", {"status": "review", "review_reason": "Human requested verification"}
        )
        before = self.store.events()
        self.assertFalse(self.store.complete_processing(source["id"], [candidate]))
        self.store.set_processing(source["id"], "deferred", "Late worker failed")
        self.assertEqual(self.store.sources()[0]["processing"], "complete")
        self.assertEqual(self.store.claims(), [])
        self.assertEqual(
            self.store.claims(False)[0]["review_reason"], "Human requested verification"
        )
        self.assertEqual(self.store.events(), before)

    def test_complete_processing_is_atomic_and_rejects_revised_source(self):
        source = self.store.capture("Atlas console is cobalt.", "atlas")
        candidate = {"id": "atlas-claim", "text": "Invented", "start": 0, "end": 24}
        with self.assertRaises(ValueError):
            self.store.complete_processing(source["id"], [candidate])
        self.assertEqual(self.store.sources()[0]["processing"], "pending")
        self.assertEqual(self.store.claims(), [])
        self.store.capture("Atlas console is amber.", "atlas")
        self.assertFalse(self.store.complete_processing(source["id"], []))


if __name__ == "__main__":
    unittest.main()
