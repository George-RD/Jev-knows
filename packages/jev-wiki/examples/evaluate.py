"""Compare lexical recall with live JEV reranking on one accepted synthetic corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Run directly from a checkout without importing the parent Cognee package.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from jev_wiki import __version__
from jev_wiki.engine import RUBRIC_VERSION, Engine
from jev_wiki.provider import JevProvider, ProviderError


class MeasuredJevProvider(JevProvider):
    """Observe the real provider without changing questions, responses, or transport."""

    def __init__(self, **options):
        super().__init__(**options)
        self.observations: list[dict] = []
        self.observed_models: set[str] = set()

    def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        started = time.perf_counter()
        before = dict(self.telemetry)
        succeeded = False
        try:
            answers = super().ask(state, questions)
            succeeded = True
            if self.last_model:
                self.observed_models.add(self.last_model)
            return answers
        finally:
            self.observations.append(
                {
                    "question_count": len(questions),
                    "ranking_candidates_offered": sum(
                        name.startswith("rank_") and question.get("type") == "score"
                        for name, question in questions.items()
                    ),
                    "state_bytes": len(state.encode("utf-8")),
                    "elapsed_seconds": time.perf_counter() - started,
                    "succeeded": succeeded,
                    "response_model": self.last_model if succeeded else None,
                    "telemetry": counter_delta(before, self.telemetry),
                }
            )


def counter_delta(before: dict, after: dict) -> dict:
    return {key: value - before.get(key, 0) for key, value in after.items()}


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def score_result(result: dict, query: dict, source_keys: dict[str, str]) -> dict:
    retrieved = list(dict.fromkeys(source_keys[item["source_id"]] for item in result["items"]))
    expected = set(query["expected_source_keys"])
    hits = len(expected.intersection(retrieved))
    recall = hits / len(expected) if expected else None
    empty_correct = not retrieved if not expected else None
    conflict_pair = set(query.get("expected_conflict_pair", []))
    linked_pairs = set()
    claim_sources = {item["id"]: source_keys[item["source_id"]] for item in result["items"]}
    for item in result["items"]:
        for relation in item.get("conflicts", []):
            target_source = claim_sources.get(relation.get("target"))
            if target_source:
                linked_pairs.add(frozenset((source_keys[item["source_id"]], target_source)))
    return {
        "mode": result["mode"],
        "degraded": result["degraded"],
        "lexical_candidate_count": result["candidate_count"],
        "retrieved_source_keys": retrieved,
        "returned_claim_count": len(result["items"]),
        "recall_at_k": recall,
        "expected_empty_correct": empty_correct,
        "query_score_at_k": recall if expected else float(empty_correct),
        "all_expected_sources_retrieved": expected.issubset(retrieved) if expected else None,
        "conflict_pair_retrieved": conflict_pair.issubset(retrieved) if conflict_pair else None,
        "conflict_pair_linked": (
            frozenset(conflict_pair) in linked_pairs if conflict_pair else None
        ),
    }


def summary(rows: list[dict], mode: str) -> dict:
    available = [row[mode] for row in rows if row[mode]["quality_metrics_valid"]]
    return {
        "valid_queries": len(available),
        "invalid_queries": len(rows) - len(available),
        "mean_recall_at_k_answerable_only": mean_or_none(
            [row["recall_at_k"] for row in available if row["recall_at_k"] is not None]
        ),
        "expected_empty_accuracy": mean_or_none(
            [
                float(row["expected_empty_correct"])
                for row in available
                if row["expected_empty_correct"] is not None
            ]
        ),
        "mean_query_score_at_k_including_empty": mean_or_none(
            [row["query_score_at_k"] for row in available]
        ),
        "elapsed_seconds_total": sum(row[mode]["elapsed_seconds"] for row in rows),
    }


def evaluate(root: Path, args: argparse.Namespace, provider: MeasuredJevProvider) -> dict:
    corpus_path = Path(__file__).with_name("eval-corpus.json")
    corpus_bytes = corpus_path.read_bytes()
    corpus = json.loads(corpus_bytes)
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "status": "incomplete",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus["name"],
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "package_version": __version__,
        "rubric_version": RUBRIC_VERSION,
        "python_version": platform.python_version(),
        "implementation_sha256": {
            name: hashlib.sha256((PACKAGE_ROOT / "jev_wiki" / name).read_bytes()).hexdigest()
            for name in ("engine.py", "provider.py", "store.py")
        },
        "configured_model": provider.model,
        "cache_enabled": False,
        "root": str(root),
        "root_retained": args.root is not None,
        "k": args.k,
        "max_chars": 20_000,
        "source_count": len(corpus["sources"]),
        "query_count": len(corpus["queries"]),
        "ingestion": [],
        "queries": [],
        "semantic_metrics": None,
        "limitations": [
            "Tiny developer-authored synthetic set; not evidence of production quality or superiority.",
            "Queries and labels were fixed before running, but are not an independent benchmark.",
            "Both retrieval modes share the same live-ingested accepted claims; rejected facts count as misses.",
            "JEV only ranks offered lexical candidates and cannot recover zero-overlap evidence.",
            "Durations include local work and network latency; token counters are not a cost invoice.",
            "No answer generation, harness injection, concurrent ingestion, or long-term memory benchmark.",
        ],
    }
    engine = Engine(root, provider=provider)
    try:
        phase_started = time.perf_counter()
        before = dict(provider.telemetry)
        for source in corpus["sources"]:
            result = engine.ingest(
                source["text"],
                source["source_key"],
                title=source["title"],
                metadata={"role": "document", "synthetic": True},
            )
            report["ingestion"].append({"source_key": source["source_key"], **result})
            if result["status"] != "complete":
                report["failure"] = "Live ingestion was deferred; no semantic comparison was run."
                return report
        report["ingestion_elapsed_seconds"] = time.perf_counter() - phase_started
        report["ingestion_telemetry"] = counter_delta(before, provider.telemetry)
        source_keys = {source["id"]: source["source_key"] for source in engine.store.sources()}
        accepted = engine.store.claims()
        accepted_keys = {source_keys[claim["source_id"]] for claim in accepted}
        report["accepted_claim_count"] = len(accepted)
        report["accepted_source_keys"] = sorted(accepted_keys)
        report["not_accepted_source_keys"] = sorted(
            {source["source_key"] for source in corpus["sources"]} - accepted_keys
        )

        # Maintenance observes the same accepted evidence. Labels are not supplied.
        phase_started = time.perf_counter()
        before = dict(provider.telemetry)
        report["maintenance"] = engine.maintain(max_pairs=100)
        report["maintenance_elapsed_seconds"] = time.perf_counter() - phase_started
        report["maintenance_telemetry"] = counter_delta(before, provider.telemetry)

        for query in corpus["queries"]:
            row = {
                **query,
                "expected_sources_accepted": sorted(
                    set(query["expected_source_keys"]) & accepted_keys
                ),
            }
            for label, offline in (("lexical", True), ("jev", False)):
                phase_started = time.perf_counter()
                before = dict(provider.telemetry)
                first_observation = len(provider.observations)
                result = engine.recall(
                    query["query"], limit=args.k, max_chars=20_000, offline=offline
                )
                elapsed = time.perf_counter() - phase_started
                observations = provider.observations[first_observation:]
                measurements = score_result(result, query, source_keys)
                # A zero-candidate result is a real pipeline miss/no-answer, not a live ranking.
                valid = (
                    offline or result["mode"] == "jev_reranked" or result["candidate_count"] == 0
                )
                measurements.update(
                    {
                        "elapsed_seconds": elapsed,
                        "quality_metrics_valid": valid,
                        "live_ranking_performed": result["mode"] == "jev_reranked",
                        "ranking_candidates_offered": sum(
                            item["ranking_candidates_offered"] for item in observations
                        ),
                        "telemetry": counter_delta(before, provider.telemetry),
                    }
                )
                if not valid:
                    for metric in (
                        "recall_at_k",
                        "expected_empty_correct",
                        "query_score_at_k",
                        "all_expected_sources_retrieved",
                        "conflict_pair_retrieved",
                        "conflict_pair_linked",
                    ):
                        measurements[metric] = None
                    measurements["failure"] = (
                        "Live ranking failed; lexical fallback is not a JEV score."
                    )
                row[label] = measurements
            report["queries"].append(row)
        report["semantic_metrics"] = {
            "lexical": summary(report["queries"], "lexical"),
            "jev": summary(report["queries"], "jev"),
        }
        complete = not report["maintenance"].get("deferred") and all(
            row["jev"]["quality_metrics_valid"] for row in report["queries"]
        )
        report["status"] = "complete" if complete else "incomplete"
        return report
    except (OSError, ValueError, ProviderError) as error:
        # Never include transport bodies or credential-bearing request objects.
        report["failure"] = f"Evaluation stopped: {type(error).__name__}."
        return report
    finally:
        report["elapsed_seconds_total"] = time.perf_counter() - started
        report["telemetry"] = dict(provider.telemetry)
        report["observed_response_models"] = sorted(provider.observed_models)
        report["provider_observations"] = provider.observations


def emit(report: dict, output: Path | None) -> None:
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, help="New or empty memory directory; default is temporary"
    )
    parser.add_argument("--output", type=Path, help="Write the JSON report here; default is stdout")
    parser.add_argument(
        "--model", help="JEV model override; default is the provider's pinned model"
    )
    parser.add_argument("--k", type=int, default=3, choices=range(1, 21), metavar="1..20")
    args = parser.parse_args(argv)

    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        emit(
            {
                "status": "refused",
                "reason": "Set TYPESAFE_API_KEY to run the live evaluation. No requests were made.",
                "semantic_metrics": None,
                "live_requests": 0,
            },
            args.output,
        )
        return 2
    if (
        args.root is not None
        and args.root.exists()
        and (not args.root.is_dir() or any(args.root.iterdir()))
    ):
        parser.error("--root must be a new or empty directory; existing memory is never reused")
    try:
        provider = MeasuredJevProvider(**({"model": args.model} if args.model else {}))
    except ProviderError as error:
        emit(
            {
                "status": "refused",
                "reason": str(error),
                "semantic_metrics": None,
                "live_requests": 0,
            },
            args.output,
        )
        return 2
    if args.root is None:
        with tempfile.TemporaryDirectory(prefix="jev-wiki-eval-") as temporary:
            report = evaluate(Path(temporary), args, provider)
    else:
        report = evaluate(args.root.resolve(), args, provider)
    emit(report, args.output)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
