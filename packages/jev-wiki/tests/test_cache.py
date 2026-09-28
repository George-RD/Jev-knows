"""Response-cache accounting, stats and clearing; no JEV credentials or network needed."""

import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from jev_wiki.cli import _parser, run
from jev_wiki.provider import JevProvider, cache_stats, clear_cache

from test_provider import RecordingTransport, choice

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import longmemeval  # noqa: E402


class CacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name) / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def provider(self, transport):
        return JevProvider("key", transport=transport, cache_dir=self.directory)

    def test_repeat_run_pays_nothing_and_counts_saved_tokens(self):
        first = self.provider(RecordingTransport())
        first.ask("source", {"q": choice(), "r": choice("Other?")})
        self.assertEqual(first.telemetry["cache_misses"], 1)
        self.assertEqual(first.telemetry["input_tokens"], 100)

        transport = RecordingTransport()
        second = self.provider(transport)
        second.ask("source", {"q": choice(), "r": choice("Other?")})
        self.assertEqual(transport.requests, [])
        self.assertEqual(second.telemetry["cache_hits"], 1)
        self.assertEqual(second.telemetry["cache_misses"], 0)
        self.assertEqual(second.telemetry["input_tokens"], 0)
        self.assertEqual(second.telemetry["cached_input_tokens"], 100)
        self.assertEqual(second.telemetry["cached_output_tokens"], 20)

    def test_misses_are_not_counted_without_a_cache(self):
        provider = JevProvider("key", transport=RecordingTransport())
        provider.ask("source", {"q": choice()})
        self.assertEqual(provider.telemetry["cache_misses"], 0)

    def test_stats_and_clear_touch_only_cache_entries(self):
        self.provider(RecordingTransport()).ask("one", {"q": choice()})
        self.provider(RecordingTransport()).ask("two", {"q": choice()})
        keep = self.directory / "notes.json"
        keep.write_text("{}")
        outside = Path(self._tmp.name) / ("v1-" + "0" * 64 + ".json")
        outside.write_text("{}")
        (self.directory / ("v1-" + "1" * 64 + ".json")).symlink_to(outside)

        stats = cache_stats(self.directory)
        self.assertEqual(stats["entries"], 2)
        self.assertGreater(stats["bytes"], 0)
        self.assertEqual(clear_cache(self.directory), 2)
        self.assertEqual(cache_stats(self.directory)["entries"], 0)
        self.assertTrue(keep.exists())
        self.assertTrue(outside.exists())

    def test_missing_or_symlinked_directory_is_empty(self):
        self.assertEqual(cache_stats(self.directory)["entries"], 0)
        self.assertEqual(clear_cache(self.directory), 0)
        self.provider(RecordingTransport()).ask("one", {"q": choice()})
        link = Path(self._tmp.name) / "link"
        link.symlink_to(self.directory)
        self.assertEqual(clear_cache(link), 0)
        self.assertEqual(cache_stats(self.directory)["entries"], 1)

    def test_cli_cache_command_reports_and_clears_the_wiki_cache(self):
        root = Path(self._tmp.name) / "wiki"
        self.directory = root / "cache" / "jev"
        self.provider(RecordingTransport()).ask("one", {"q": choice()})
        shown = run(_parser().parse_args(["--root", str(root), "cache"]))
        self.assertEqual((shown["entries"], shown["removed"]), (1, 0))
        cleared = run(_parser().parse_args(["--root", str(root), "cache", "--clear"]))
        self.assertEqual((cleared["entries"], cleared["removed"]), (0, 1))
        self.assertFalse((root / "state.json").exists())

    def test_harness_cache_report(self):
        telemetry = Counter(cache_hits=3, cache_misses=1, input_tokens=40, cached_input_tokens=90)
        report = longmemeval.cache_report(str(self.directory), telemetry)
        self.assertEqual(
            (report["enabled"], report["hits"], report["misses"], report["saved_input_tokens"]),
            (True, 3, 1, 90),
        )
        self.assertEqual(report["entries"], 0)
        self.assertFalse(longmemeval.cache_report(None, Counter())["enabled"])

    def test_harness_reports_cache_off_for_model_aliases(self):
        data = Path(self._tmp.name) / "empty.json"
        data.write_text("[]")
        output = Path(self._tmp.name) / "report.json"
        argv = ["longmemeval", "--data", str(data), "--output", str(output), "--workers", "1"]
        argv += ["--cache-dir", str(self.directory)]
        for model, enabled in (("jev-latest", False), ("jev-1.13.0", True)):
            with (
                patch.dict(os.environ, {"TYPESAFE_API_KEY": "fixture-key"}),
                patch.object(sys, "argv", [*argv, "--model", model]),
                patch("builtins.print"),
            ):
                self.assertEqual(longmemeval.main(), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["cache"]["enabled"], enabled)

    def test_harness_default_cache_dir_prefers_env_then_shared_folder(self):
        shared = Path(self._tmp.name) / "project-files"
        with patch.object(longmemeval, "SHARED_FOLDER", shared):
            with patch.dict(os.environ, {"JEV_WIKI_CACHE_DIR": str(self.directory)}):
                self.assertEqual(longmemeval.default_cache_dir(), self.directory)
            with patch.dict(os.environ, {"JEV_WIKI_CACHE_DIR": ""}):
                self.assertNotEqual(longmemeval.default_cache_dir().parent.name, "jev-cache")
                shared.mkdir()
                self.assertEqual(
                    longmemeval.default_cache_dir(), shared / "jev-cache" / "longmemeval"
                )

    def test_harness_default_cache_dir_honours_xdg(self):
        env = {"XDG_CACHE_HOME": self._tmp.name, "JEV_WIKI_CACHE_DIR": ""}
        missing = Path(self._tmp.name) / "no-shared-folder"
        with patch.dict(os.environ, env), patch.object(longmemeval, "SHARED_FOLDER", missing):
            self.assertEqual(
                longmemeval.default_cache_dir(),
                Path(self._tmp.name) / "jev-wiki" / "longmemeval",
            )


if __name__ == "__main__":
    unittest.main()
