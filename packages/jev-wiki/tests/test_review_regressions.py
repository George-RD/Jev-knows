"""PR #1 regressions: synthetic evidence and transport only, never live services."""

import html
import http.client
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_wiki.cli import _process_pending
from jev_wiki.engine import Engine
from jev_wiki.hooks import drain_inbox, handle_hook, hook_time_budget, queue_event
from jev_wiki.provider import JevProvider, ProviderError
from jev_wiki.store import WikiStore
from test_provider import RecordingTransport, choice
from test_races import _DecisionFixture


class ReviewRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jev-review-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def payload(self, project, text="Project Orion uses SQLite."):
        return {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "test-session",
            "cwd": str(project),
            "prompt": text,
        }

    def test_hook_injects_full_normal_paragraph_that_fits_actual_budget(self):
        project = self.directory / "project"
        project.mkdir()
        root = project / ".memory"
        # One sentence near the candidate size limit: a whole claim, not a fragment.
        text = (
            "Orion uses SQLite, " + "preserving the complete decision and its scope, " * 11
        ) + "in every release."
        engine = Engine(root, _DecisionFixture())
        engine.ingest(text, "orion-decision")
        result = handle_hook(
            root, self.payload(project, "What does Orion use?"), project_root=project
        )
        self.assertIn("hookSpecificOutput", result)
        context = result["hookSpecificOutput"]["additionalContext"]
        self.assertIn(text, context)
        self.assertIn("#chars=0-", context)
        self.assertLessEqual(len(context), 6000)

    def test_worker_rejects_foreign_project_event_before_ingestion(self):
        projects = [self.directory / name for name in ("first", "second")]
        for project in projects:
            project.mkdir()
            handle_hook(
                project / ".memory", self.payload(project, project.name), project_root=project
            )
        foreign = next((projects[0] / ".memory/inbox/hooks").glob("*.json"))
        root = projects[1] / ".memory"
        shutil.copyfile(foreign, root / "inbox/hooks" / foreign.name)
        engine = Engine(root)
        result = drain_inbox(root, engine)
        self.assertEqual(result["captured"], 1)
        self.assertEqual(result["failed"], 1)
        sources = engine.store.sources()
        self.assertEqual(len(sources), 1)
        self.assertEqual(engine.store.read_source(sources[0]["id"]), "second")
        self.assertTrue((root / "inbox/hooks" / foreign.name).exists())

    def test_worker_rejects_mismatched_event_role(self):
        project = self.directory / "project"
        project.mkdir()
        root = project / ".memory"
        handle_hook(root, self.payload(project), project_root=project)
        original = next((root / "inbox/hooks").glob("*.json"))
        record = json.loads(original.read_text())
        original.unlink()
        queue_event(
            root, record["project_id"], record["session_id"], "Stop", "Assistant proposal", "user"
        )
        engine = Engine(root)
        result = drain_inbox(root, engine)
        self.assertEqual(result["captured"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(engine.store.sources(), [])

    def test_cache_does_not_persist_unvalidated_response_fields(self):
        sentinel = "synthetic-private-echo-not-for-cache"

        def extra_fields(body):
            body["debug"] = {"state": sentinel, "Authorization": sentinel}
            body["answers"]["q"]["explanation"] = sentinel
            body["usage"]["request"] = sentinel

        cache = self.directory / "cache"
        transport = RecordingTransport([extra_fields])
        provider = JevProvider("fixture-key", transport=transport, cache_dir=cache)
        expected = provider.ask("source", {"q": choice()})
        for path in cache.glob("*.json"):
            self.assertNotIn(sentinel, path.read_text())
        self.assertEqual(provider.ask("source", {"q": choice()}), expected)
        self.assertEqual(len(transport.requests), 1)

    def test_truncated_http_body_retries_and_recovers(self):
        transport = RecordingTransport([http.client.IncompleteRead(b"private partial body", 100)])
        provider = JevProvider("fixture-key", transport=transport, sleep=lambda _: None)
        self.assertEqual(provider.ask("source", {"q": choice()})["q"]["value"], "supported")
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(provider.telemetry["retries"], 1)

    def test_truncated_http_body_exhaustion_is_redacted_provider_error(self):
        transport = RecordingTransport(
            [http.client.IncompleteRead(b"private", 100) for _ in range(3)]
        )
        provider = JevProvider("fixture-key", transport=transport, sleep=lambda _: None)
        with self.assertRaises(ProviderError) as caught:
            provider.ask("source", {"q": choice()})
        self.assertNotIn("private", str(caught.exception))
        self.assertEqual(len(transport.requests), 3)
        self.assertEqual(provider.telemetry["failures"], 1)

    def test_extreme_response_number_is_validation_error_not_overflow(self):
        for field in ("confidence", "probabilities"):
            with self.subTest(field=field):

                def huge_number(body):
                    if field == "confidence":
                        body["answers"]["q"][field] = 10**400
                    else:
                        body["answers"]["q"][field]["supported"] = 10**400

                provider = JevProvider("fixture-key", transport=RecordingTransport([huge_number]))
                with self.assertRaises(ProviderError):
                    provider.ask("source", {"q": choice()})

    def test_escaped_source_batches_fit_real_adapter_wire_budget(self):
        transport = RecordingTransport()
        provider = JevProvider("fixture-key", transport=transport)
        engine = Engine(self.directory / "memory", provider)
        text = "\n\n".join(f"Paragraph {i}: " + "\x01" * 1400 for i in range(4))
        result = engine.ingest(text, "escaped-source")
        self.assertEqual(result["status"], "complete")
        self.assertGreater(len(transport.requests), 1)
        for request, _, payload in transport.requests:
            self.assertLessEqual(len(request.data), provider.MAX_REQUEST_BYTES)
            for key, question in payload["questions"].items():
                single = {**payload, "questions": {key: question}}
                size = len(json.dumps(single, ensure_ascii=False, separators=(",", ":")).encode())
                self.assertLessEqual(size, provider.MAX_STATE_QUESTION_BYTES)

    def test_pending_worker_continues_after_source_is_forgotten_mid_run(self):
        engine = Engine(self.directory / "memory")
        for key in ("alpha", "beta", "gamma"):
            engine.store.capture(f"Project {key} uses SQLite.", key)
        sources = engine.store.sources()
        removed = sources[1]
        engine.provider = _DecisionFixture(
            before_answer=lambda: engine.store.forget(removed["source_key"])
        )
        result = _process_pending(engine, 10)
        self.assertEqual(
            [item["status"] for item in result["results"]], ["complete", "cancelled", "complete"]
        )
        self.assertTrue(
            all(source["processing"] == "complete" for source in engine.store.sources())
        )

    def test_offline_processing_returns_cancelled_on_concurrent_forget(self):
        engine = Engine(self.directory / "memory")
        source = engine.store.capture("Project Orion uses SQLite.", "orion")
        original = engine.store.set_processing

        def forget_first(*args, **kwargs):
            WikiStore(engine.root).forget("orion")
            return original(*args, **kwargs)

        with patch.object(engine.store, "set_processing", side_effect=forget_first):
            result = engine.process(source["id"])
        self.assertEqual(result["status"], "cancelled")

    def test_hook_uses_actual_escape_size_and_preserves_complete_citations(self):
        project = self.directory / "project"
        project.mkdir()
        root = project / ".memory"
        engine = Engine(root, _DecisionFixture())
        # Each large claim escapes to about 3000 characters: both cannot fit.
        large = ["Orion " + "&" * 590, "Orion also " + "&" * 585]
        small = "Orion uses <SQLite> & keeps a cited audit trail."
        engine.ingest(large[0], "escaped-large-one")
        engine.ingest(large[1], "escaped-large-two")
        engine.ingest(small, "escaped-fits")
        result = handle_hook(root, self.payload(project, "Orion?"), project_root=project)
        context = result["hookSpecificOutput"]["additionalContext"]
        self.assertIn(html.escape(small, quote=False), context)
        included = sum(html.escape(text, quote=False) in context for text in large)
        self.assertEqual(included, 1)
        # Every citation carries its complete quote; nothing was truncated to fit.
        self.assertEqual(context.count("#chars="), 2)
        self.assertEqual(context.count("</untrusted_memory_evidence>"), 1)
        self.assertLessEqual(len(context), 6000)

    def test_worker_keeps_invalid_or_unbound_events_for_inspection(self):
        project = self.directory / "project"
        project.mkdir()
        root = project / ".memory"
        handle_hook(root, self.payload(project), project_root=project)
        path = next((root / "inbox/hooks").glob("*.json"))
        original = json.loads(path.read_text())
        path.unlink()
        for event, role, text in (
            ("Unsupported", "user", "Project Orion uses SQLite."),
            ("UserPromptSubmit", "assistant", "Assistant proposal"),
            ("UserPromptSubmit", "user", " "),
            ("UserPromptSubmit", "user", "x" * 24_001),
        ):
            queue_event(root, original["project_id"], original["session_id"], event, text, role)
        engine = Engine(root)
        self.assertEqual(drain_inbox(root, engine)["failed"], 4)
        self.assertEqual(engine.store.sources(), [])
        self.assertEqual(len(list(path.parent.glob("*.json"))), 4)
        for event_path in path.parent.glob("*.json"):
            event_path.unlink()
        queue_event(
            root,
            original["project_id"],
            original["session_id"],
            "UserPromptSubmit",
            "Valid message",
            "user",
        )
        (root / "hook-project.json").unlink()
        self.assertEqual(drain_inbox(root, engine)["failed"], 1)
        self.assertEqual(engine.store.sources(), [])

    def test_cache_sanitizes_existing_valid_entries_on_read(self):
        cache = self.directory / "cache"
        transport = RecordingTransport()
        provider = JevProvider("fixture-key", transport=transport, cache_dir=cache)
        expected = provider.ask("source", {"q": choice()})
        path = next(cache.glob("*.json"))
        raw = json.loads(path.read_text())
        raw["debug"] = "private-old-echo"
        path.write_text(json.dumps(raw))
        self.assertEqual(provider.ask("source", {"q": choice()}), expected)
        self.assertNotIn("private-old-echo", path.read_text())
        self.assertEqual(len(transport.requests), 1)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX special files required")
    def test_cache_fifo_is_a_nonblocking_miss(self):
        cache = self.directory / "cache"
        transport = RecordingTransport()
        provider = JevProvider("fixture-key", transport=transport, cache_dir=cache)
        expected = provider.ask("source", {"q": choice()})
        path = next(cache.glob("*.json"))
        path.unlink()
        os.mkfifo(path)
        with hook_time_budget(0.5):
            self.assertEqual(provider.ask("source", {"q": choice()}), expected)
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(provider.telemetry["cache_errors"], 1)
        self.assertTrue(path.is_file())

    def test_cache_symlink_does_not_read_or_modify_its_target(self):
        cache = self.directory / "cache"
        transport = RecordingTransport()
        provider = JevProvider("fixture-key", transport=transport, cache_dir=cache)
        expected = provider.ask("source", {"q": choice()})
        path = next(cache.glob("*.json"))
        target = self.directory / "outside.json"
        original = path.read_bytes()
        target.write_bytes(original)
        path.unlink()
        path.symlink_to(target)
        self.assertEqual(provider.ask("source", {"q": choice()}), expected)
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse(path.is_symlink())

    def test_invalid_current_source_errors_are_not_hidden_by_pending_worker(self):
        engine = Engine(self.directory / "memory")
        engine.store.capture("Project Orion uses SQLite.", "orion")
        with patch.object(engine, "process", side_effect=ValueError("invalid current source")):
            with self.assertRaisesRegex(ValueError, "invalid current source"):
                _process_pending(engine, 10)

    def test_failure_deferral_returns_cancelled_when_source_changes_during_write(self):
        engine = Engine(self.directory / "memory", _DecisionFixture(fail=True))
        source = engine.store.capture("Project Orion uses SQLite.", "orion")
        original = engine.store.set_processing

        def forget_first(*args, **kwargs):
            WikiStore(engine.root).forget("orion")
            return original(*args, **kwargs)

        with patch.object(engine.store, "set_processing", side_effect=forget_first):
            result = engine.process(source["id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(engine.store.claims(), [])

    def test_extreme_custom_confidence_is_not_promoted(self):
        provider = _DecisionFixture()
        original = provider.ask

        def huge_confidence(state, questions):
            answers = original(state, questions)
            for answer in answers.values():
                answer["confidence"] = 10**400
            return answers

        provider.ask = huge_confidence
        engine = Engine(self.directory / "memory", provider)
        result = engine.ingest("Project Orion uses SQLite.", "orion")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["active"], 0)
        self.assertEqual(result["review"], 1)

    def test_extreme_store_confidence_and_timeout_fail_validation(self):
        engine = Engine(self.directory / "memory", _DecisionFixture())
        engine.ingest("Project Orion uses SQLite.", "orion")
        claim = engine.store.claims()[0]
        with self.assertRaises(ValueError):
            engine.store.update_claim(claim["id"], {"confidence": 10**400})
        with self.assertRaises(ProviderError):
            JevProvider("fixture-key", timeout=10**400)
        self.assertEqual(engine.store.claims()[0]["confidence"], 0.99)

    def test_wire_bounded_batches_preserve_global_ids_and_exact_quotes(self):
        class CapturingFixture(_DecisionFixture):
            def __init__(self):
                super().__init__()
                self.calls = []

            def ask(self, state, questions):
                self.calls.append((state, questions))
                return super().ask(state, questions)

        provider = CapturingFixture()
        engine = Engine(self.directory / "memory", provider)
        text = "\n\n".join(f"Paragraph {i}: " + "\x01" * 1400 for i in range(9))
        result = engine.ingest(text, "escaped-source")
        # Each 1400-character paragraph is cut into three bounded candidates.
        self.assertEqual(result["claims"], 27)
        observed = []
        for state, questions in provider.calls:
            indices = list(json.loads(state)["candidates"])
            observed.extend(indices)
            self.assertEqual(
                set(questions),
                {f"{kind}_{i}" for i in indices for kind in ("keep", "kind", "topic")},
            )
        self.assertEqual(observed, [str(i) for i in range(27)])
        for claim in engine.store.claims():
            self.assertEqual(claim["text"], text[claim["start"] : claim["end"]])
        self.assertEqual(engine.store.lint(), [])


if __name__ == "__main__":
    unittest.main()
