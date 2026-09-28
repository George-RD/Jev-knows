"""End-to-end question answering on LongMemEval_S: does recalled memory answer the question?

The retrieval harnesses (``longmemeval.py``, ``longmemeval_offline.py``) only ask whether a
labelled session reached the top k. This one asks what an agent needs: a reader model
answers each question from what jev-wiki recalls, and a judge model grades the answer
with LongMemEval's own per-type judge prompts (Wu et al., ICLR 2025, ``evaluate_qa.py``).

Intake is the offline keep-everything stub from ``longmemeval_offline.py`` and recall runs
with ``offline=True``, so no TypeSafe credits are spent. It measures recall plus answering
with an ideal intake, not the live keep policy. Only user turns are ingested (as in the
other harnesses), so ``single-session-assistant`` questions are expected to fail.

Context modes:

- ``wiki``: the claims ``Engine.recall(offline=True)`` returns (up to 20, 20k chars), each
  prefixed with the date of the chat it came from. An agent's memory knows when a note was
  captured; LongMemEval's temporal questions are unanswerable without it.
  ``--aggregate-limit N`` lets counting and date questions recall up to N claims. Each
  chat is ingested with its date and recall runs as of the question's date, so "two
  weeks ago" resolves the way it would have that day.
  ``--live`` instead runs the shipped pipeline: JEV intake with the shipped keep gate, then
  recall with the JEV rerank at ``--min-relevance``. That spends TypeSafe credits, so it
  goes through the same shared response cache as ``longmemeval.py`` (``--cache-dir``);
  a question that intake could not complete is an error row, not a wrong answer.
- ``oracle``: the user turns of the labelled evidence sessions, dated. The ceiling for a
  store that keeps only user turns; the gap to ``wiki`` is what recall and claim
  splitting lose.
- ``none``: no memory. The floor: what the reader guesses.

Abstention questions (``_abs``) are included by default because not inventing an answer
is part of being good memory; ``--no-abstention`` drops them.

Reader and judge run on any Ollama-compatible ``/api/chat`` endpoint. The default is
Ollama Cloud (``OLLAMA_API_KEY``, sent as a bearer token when set). LongMemEval's
published numbers use GPT-4o as the judge, so compare against them with care; compare
runs of this harness with each other using the same models.

    python packages/jev-wiki/examples/longmemeval_qa.py --data longmemeval_s_cleaned.json \
        --per-type 5 --output /tmp/lme-qa.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import statistics
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jev_wiki import __version__  # noqa: E402
from jev_wiki.engine import MIN_RELEVANCE, Engine  # noqa: E402
from jev_wiki.provider import JevProvider, ProviderError  # noqa: E402
from longmemeval import default_cache_dir, score_sessions, select, session_text  # noqa: E402
from longmemeval_offline import KeepEverything, _embedder  # noqa: E402

MODES = ("wiki", "oracle", "none")

READER_PROMPT = (
    "I will give you notes from your past chats with a user, each with the date of its "
    "chat. Please answer the question based on the relevant notes.\n\n\n"
    "Notes:\n\n{notes}\n\nCurrent Date: {date}\nQuestion: {question}\nAnswer:"
)

# LongMemEval's judge prompts, verbatim apart from line wrapping.
_JUDGE_TAIL = "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}"
_JUDGE_BASE = (
    "I will give you a question, a correct answer, and a response from a model. Please "
    "answer yes if the response contains the correct answer. Otherwise, answer no. If the "
    "response is equivalent to the correct answer or contains all the intermediate steps "
    "to get the correct answer, you should also answer yes. If the response only contains "
    "a subset of the information required by the answer, answer no."
)
JUDGE_PROMPTS = {
    "default": _JUDGE_BASE
    + _JUDGE_TAIL
    + "\n\nIs the model response correct? Answer yes or no only.",
    "temporal-reasoning": _JUDGE_BASE
    + " In addition, do not penalize off-by-one errors for the number of days. If the "
    "question asks for the number of days/weeks/months, etc., and the model makes "
    "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
    "response is still correct."
    + _JUDGE_TAIL
    + "\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": _JUDGE_BASE
    + " If the response contains some previous information along with an updated answer, "
    "the response should be considered as correct as long as the updated answer is the "
    "required answer." + _JUDGE_TAIL + "\n\nIs the model response correct? Answer yes or no only.",
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a "
        "response from a model. Please answer yes if the response satisfies the desired "
        "response. Otherwise, answer no. The model does not need to reflect all the points "
        "in the rubric. The response is correct as long as it recalls and utilizes the "
        "user's personal information correctly.\n\nQuestion: {question}\n\nRubric: {answer}"
        "\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no "
        "only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a "
        "model. Please answer yes if the model correctly identifies the question as "
        "unanswerable. The model could say that the information is incomplete, or some "
        "other information is given but the asked information is not.\n\nQuestion: "
        "{question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\nDoes the model "
        "correctly identify the question as unanswerable? Answer yes or no only."
    ),
}

_CONFIG: dict = {}
ATTEMPTS = 7
# URLError and timeouts are OSErrors; dropped connections and cut-off bodies are too, or
# HTTPExceptions. One of them must cost a row, never the run.
_CALL_ERRORS = (OSError, http.client.HTTPException, KeyError, json.JSONDecodeError)


class IntakeIncomplete(RuntimeError):
    """Live intake left a session unprocessed; the store would turn it into a miss."""


def chat(model: str, prompt: str) -> str:
    """One non-streaming /api/chat call, retried on transient failures."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"
    request = urllib.request.Request(
        _CONFIG["endpoint"], data=json.dumps(body).encode(), headers=headers
    )
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)["message"]["content"].strip()
        except _CALL_ERRORS as exc:
            status = getattr(exc, "code", None)
            if attempt == ATTEMPTS - 1 or (status is not None and status < 500 and status != 429):
                raise
            # Rate limits and gateway errors clear in tens of seconds, not two.
            time.sleep(min(60, 2 ** (attempt + 1)) + random.random())
    raise AssertionError("unreachable")


def judge_prompt(item: dict, response: str) -> str:
    kind = "abstention" if item["question_id"].endswith("_abs") else item["question_type"]
    template = JUDGE_PROMPTS.get(kind, JUDGE_PROMPTS["default"])
    return template.format(question=item["question"], answer=item["answer"], response=response)


def iso_date(longmemeval_date: str) -> str:
    """'2023/05/30 (Tue) 23:40' -> '2023-05-30'."""
    return longmemeval_date.split(" ", 1)[0].replace("/", "-")


def wiki_notes(item: dict) -> tuple[str, dict]:
    """Recall from a fresh wiki; return dated notes and session recall.

    Offline, intake keeps everything and recall is local. Live, JEV decides intake and
    reranks the recall shortlist.
    """
    expected_ids = set(item["answer_session_ids"])
    date_of, expected = {}, set()
    live = _CONFIG.get("live")
    provider = (
        JevProvider(model=_CONFIG["jev_model"], cache_dir=_CONFIG["cache_dir"])
        if live
        else KeepEverything()
    )
    statuses: dict[str, int] = defaultdict(int)
    with tempfile.TemporaryDirectory(prefix="jev-lme-qa-") as root:
        engine = Engine(root, provider, embedder=_embedder())
        for index, (sid, date, session) in enumerate(
            zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"])
        ):
            key = f"s{index:03d}"
            date_of[key] = date
            if sid in expected_ids:
                expected.add(key)
            text = session_text(session)
            if text:
                metadata = {"role": "user", "date": iso_date(date)}
                result = engine.ingest(text, source_key=key, title=key, metadata=metadata)
                statuses[result["status"]] += 1
        if live and set(statuses) - {"complete"}:
            raise IntakeIncomplete(f"ingestion incomplete: {dict(statuses)}")
        if not live:
            for claim in engine.store.claims(active_only=False):
                if claim["status"] != "active":
                    engine.store.update_claim(claim["id"], {"status": "active"})
        key_of = {s["id"]: s["source_key"] for s in engine.store.sources()}
        active = len(engine.store.claims())
        result = engine.recall(
            item["question"],
            limit=20,
            max_chars=20_000,
            offline=not live,
            aggregate_limit=_CONFIG.get("aggregate_limit"),
            as_of=iso_date(item["question_date"]),
            min_relevance=MIN_RELEVANCE if not live else _CONFIG["min_relevance"],
        )
    keys = [key_of[i["source_id"]] for i in result["items"]]
    notes = "\n".join(
        f"[{n}] (chat date: {date_of[key]}) {json.dumps(i['text'], ensure_ascii=False)}"
        for n, (key, i) in enumerate(zip(keys, result["items"]), 1)
    )
    retrieval = score_sessions(keys, expected) if expected else {}
    stats = {
        "claims": len(result["items"]),
        "aggregate": result["aggregate"],
        "time_window": result["time_window"],
        "recall_any@10": retrieval.get("recall_any@10"),
    }
    if live:
        stats.update(
            claims_active=active,
            recall_mode=result["mode"],
            degraded=result["degraded"],
            telemetry=dict(provider.telemetry),
        )
    return notes, stats


def oracle_notes(item: dict) -> str:
    evidence = set(item["answer_session_ids"])
    blocks = [
        f"### Chat on {date}\n{session_text(session)}"
        for sid, date, session in zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]
        )
        if sid in evidence and session_text(session)
    ]
    return "\n\n".join(blocks)


def run_question(item: dict) -> dict:
    row = {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "abstention": item["question_id"].endswith("_abs"),
        "question": item["question"],
        "answer": str(item["answer"]),
    }
    mode = _CONFIG["mode"]
    started = time.perf_counter()
    if mode == "wiki":
        try:
            notes, stats = wiki_notes(item)
        except (IntakeIncomplete, ProviderError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            return row
        row.update(stats)
    elif mode == "oracle":
        notes = oracle_notes(item)
    else:
        notes = ""
    row["recall_seconds"] = round(time.perf_counter() - started, 2)
    row["notes_chars"] = len(notes)
    prompt = READER_PROMPT.format(
        notes=notes or "(none)", date=item["question_date"], question=item["question"]
    )
    try:
        row["hypothesis"] = chat(_CONFIG["reader"], prompt)
        verdict = chat(
            _CONFIG["judge"], judge_prompt({**item, "answer": row["answer"]}, row["hypothesis"])
        )
        row["judge_raw"] = verdict[:200]
        row["correct"] = "yes" in verdict.lower()
    except _CALL_ERRORS as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def _init(config: dict) -> None:
    _CONFIG.update(config)


def summarize(rows: list[dict]) -> dict:
    def accuracy(subset: list[dict]) -> float | None:
        graded = [r["correct"] for r in subset if "correct" in r]
        return round(statistics.mean(graded), 3) if graded else None

    by_type = defaultdict(list)
    for row in rows:
        by_type["abstention" if row["abstention"] else row["question_type"]].append(row)
    out = {
        "questions": len(rows),
        "errors": sum("error" in r for r in rows),
        "accuracy": accuracy(rows),
        "accuracy_answerable": accuracy([r for r in rows if not r["abstention"]]),
        "by_type": {k: {"n": len(v), "accuracy": accuracy(v)} for k, v in sorted(by_type.items())},
    }
    telemetry = defaultdict(int)
    for row in rows:
        for key, value in row.get("telemetry", {}).items():
            telemetry[key] += value
    if telemetry:
        out["jev"] = {
            "paid_requests": telemetry["requests"],
            "cache_hits": telemetry["cache_hits"],
            "input_tokens": telemetry["input_tokens"],
            "degraded_recalls": sum(bool(r.get("degraded")) for r in rows),
            "claims_returned_mean": round(
                statistics.mean(r["claims"] for r in rows if "claims" in r), 1
            ),
        }
    retrieved = [r for r in rows if r.get("recall_any@10") is not None and "correct" in r]
    if retrieved:
        hit = [r for r in retrieved if r["recall_any@10"]]
        miss = [r for r in retrieved if not r["recall_any@10"]]
        # Splits wrong answers into "memory never surfaced it" and "surfaced but not used".
        out["accuracy_when_session_recalled@10"] = {"n": len(hit), "accuracy": accuracy(hit)}
        out["accuracy_when_session_missed@10"] = {"n": len(miss), "accuracy": accuracy(miss)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="longmemeval_s_cleaned.json")
    parser.add_argument("--per-type", type=int, default=5, help="0 for every question")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=MODES, default="wiki")
    parser.add_argument("--no-abstention", action="store_true")
    parser.add_argument("--reader", default="gpt-oss:120b")
    parser.add_argument("--judge", default="glm-5.3")
    parser.add_argument("--endpoint", default="https://ollama.com/api/chat")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--aggregate-limit",
        type=int,
        help="wiki mode: recall up to this many claims for counting and date questions",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="wiki mode: JEV intake and rerank (paid; cached) instead of offline recall",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=MIN_RELEVANCE,
        help="--live: JEV rerank cut (default: the shipped %(default)s)",
    )
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="--live: JEV response cache shared with longmemeval.py (default: %(default)s)",
    )
    parser.add_argument("--output", help="write the full report (summary and rows) here")
    parser.add_argument(
        "--resume", help="an earlier --output report: keep its graded rows, rerun the rest"
    )
    args = parser.parse_args()

    if args.live and args.mode != "wiki":
        parser.error("--live needs --mode wiki")
    items = json.loads(Path(args.data).read_text())
    per_type = args.per_type or len(items)
    items = select(items, per_type, args.seed, include_abstention=not args.no_abstention)
    config = {
        "mode": args.mode,
        "reader": args.reader,
        "judge": args.judge,
        "endpoint": args.endpoint,
        "aggregate_limit": args.aggregate_limit,
        "live": args.live,
        "min_relevance": args.min_relevance if args.live else None,
        "jev_model": args.jev_model if args.live else None,
        "cache_dir": str(args.cache_dir) if args.live else None,
    }
    done = {}
    if args.resume:
        earlier = json.loads(Path(args.resume).read_text())
        resumed = {**config, "embedding_model": os.environ.get("JEV_WIKI_EMBEDDING_MODEL")}
        del resumed["cache_dir"]  # where responses are cached does not change them
        # Reports from before --live ran offline.
        before_live = {"live": False}
        if {k: earlier["config"].get(k, before_live.get(k)) for k in resumed} != resumed:
            parser.error("--resume report was made with a different mode, models or embedder")
        done = {r["question_id"]: r for r in earlier["rows"] if "correct" in r}
    todo = [item for item in items if item["question_id"] not in done]
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(config,)) as pool:
        for count, row in enumerate(pool.map(run_question, todo, chunksize=1), 1):
            done[row["question_id"]] = row
            if count % 25 == 0:
                print(f"{count}/{len(todo)} answered", file=sys.stderr, flush=True)
    rows = [done[item["question_id"]] for item in items]
    report = {
        "jev_wiki_version": __version__,
        "config": {
            **config,
            "per_type": args.per_type,
            "seed": args.seed,
            "abstention": not args.no_abstention,
            "embedding_model": os.environ.get("JEV_WIKI_EMBEDDING_MODEL"),
        },
        "seconds": round(time.perf_counter() - started, 1),
        "summary": summarize(rows),
    }
    print(json.dumps(report, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps({**report, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
