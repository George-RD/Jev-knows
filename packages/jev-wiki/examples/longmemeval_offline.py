"""Offline shortlist check on LongMemEval_S: no JEV calls, no credits.

Every sentence candidate is kept and activated (an upper bound like the live harness's
``all_stored`` policy), then ``Engine.recall(offline=True)`` is scored with session
recall. This isolates candidate generation: run it from two checkouts to compare
shortlist rankings. Keep answers come from a stub, so this says nothing about intake.
Set ``JEV_WIKI_EMBEDDING_MODEL=default`` (with the ``embed`` extra installed) to score
recall with local embedding candidates.

    python packages/jev-wiki/examples/longmemeval_offline.py \
        --data longmemeval_s_cleaned.json --per-type 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jev_wiki.embedding import from_env  # noqa: E402
from jev_wiki.engine import Engine  # noqa: E402
from longmemeval import KS, score_sessions, select, session_text  # noqa: E402


class KeepEverything:
    """Stub provider: keeps every candidate. Recall runs offline, so ranking is never asked."""

    model = "offline-keep-all-stub"

    def ask(self, state, questions):
        values = {"keep": "keep", "kind": "fact", "topic": "general"}
        return {
            name: {"value": values[name.split("_")[0]], "confidence": 1.0} for name in questions
        }


_EMBEDDER: list = []


def _embedder():
    """One embedder per worker process, from JEV_WIKI_EMBEDDING_MODEL (None when unset)."""
    if not _EMBEDDER:
        _EMBEDDER.append(from_env())
    return _EMBEDDER[0]


def run_question(item: dict) -> tuple[str, dict]:
    expected_ids = set(item["answer_session_ids"])
    with tempfile.TemporaryDirectory(prefix="jev-lme-offline-") as root:
        engine = Engine(root, KeepEverything(), embedder=_embedder())
        expected = set()
        for index, (sid, session) in enumerate(
            zip(item["haystack_session_ids"], item["haystack_sessions"])
        ):
            key = f"s{index:03d}"
            if sid in expected_ids:
                expected.add(key)
            text = session_text(session)
            if text:
                engine.ingest(text, source_key=key, title=key, metadata={"role": "user"})
        for claim in engine.store.claims(active_only=False):
            if claim["status"] != "active":
                engine.store.update_claim(claim["id"], {"status": "active"})
        key_of = {s["id"]: s["source_key"] for s in engine.store.sources()}
        result = engine.recall(item["question"], limit=20, max_chars=20_000, offline=True)
        ranked = [key_of[i["source_id"]] for i in result["items"]]
    return item["question_type"], score_sessions(ranked, expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="longmemeval_s_cleaned.json")
    parser.add_argument("--per-type", type=int, default=5, help="0 for every question")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    items = json.loads(Path(args.data).read_text())
    items = select(items, args.per_type or len(items), args.seed, include_abstention=False)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_question, items, chunksize=2))
    by_type = defaultdict(list)
    for kind, row in rows:
        by_type[kind].append(row)

    def mean(subset, key):
        return round(statistics.mean(r[key] for r in subset), 3)

    everything = [row for _, row in rows]
    report = {
        "questions": len(rows),
        **{f"recall_any@{k}": mean(everything, f"recall_any@{k}") for k in KS},
        "recall_all@10": mean(everything, "recall_all@10"),
        "by_type_recall_any@10": {k: mean(v, "recall_any@10") for k, v in sorted(by_type.items())},
    }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
