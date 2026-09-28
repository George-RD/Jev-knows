"""Answer-sentence recall on LongMemEval_S: does recall return the sentences that answer?

Session recall asks whether a labelled chat reached the top k, and the QA harness
(``longmemeval_qa.py``) asks a reader model to answer. This sits between them and costs
nothing per run: each question's evidence turns are split into sentences the way intake
splits them, and a labels file names the sentences the answer depends on. A run scores
how many of those sentences ``Engine.recall(offline=True)`` returns, and how many it
misses although another sentence from the same chat came back.

LongMemEval marks answer turns (``has_answer``), not sentences, so the labels come from
one reader call per question (``--label``, on Ollama like the QA harness). The labels
from 2026-09-28 are in the project's shared folder
(``benchmarks/longmemeval-answer-sentences-2026-09-28.jsonl``). Intake is the offline
keep-everything stub; ``--wikis DIR`` keeps each question's wiki, so later runs only
recall (about 40 s for all 500 on four cores, against about an hour to build them).

    python packages/jev-wiki/examples/longmemeval_sentences.py \
        --data longmemeval_s_cleaned.json --labels answer-sentences.jsonl \
        --wikis /tmp/lme-wikis --aggregate-limit 40 --neighbours
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jev_wiki.engine import Engine, candidate_spans  # noqa: E402
from longmemeval import session_text  # noqa: E402
from longmemeval_offline import KeepEverything, _embedder  # noqa: E402
from longmemeval_qa import _CALL_ERRORS, _CONFIG, _init, chat, iso_date  # noqa: E402

LABEL_PROMPT = """A user asked a question about their own past chats. Below are numbered \
sentences the user wrote in the chats that contain the answer.

Question (asked on {date}): {question}
Correct answer: {answer}

Sentences:
{sentences}

Which sentences state a fact the correct answer depends on (the item, event, number, \
date or detail itself, not just the topic)? Reply with JSON only: {{"needed": [numbers]}}"""


def evidence_sentences(item: dict) -> list[dict]:
    """The sentences of the question's answer turns (user turns only, as ingested)."""
    sentences = []
    for sid, date, session in zip(
        item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]
    ):
        for turn in session:
            if turn["role"] == "user" and turn.get("has_answer"):
                for span in candidate_spans(turn["content"].strip()):
                    sentences.append({"sid": sid, "date": date, "text": span["text"]})
    return sentences


def label(item: dict) -> dict:
    sentences = evidence_sentences(item)
    row = {"question_id": item["question_id"], "sentences": sentences, "needed": []}
    if not sentences:
        return row
    listing = "\n".join(f"{i}. (chat {s['date']}) {s['text']}" for i, s in enumerate(sentences))
    prompt = LABEL_PROMPT.format(
        date=item["question_date"],
        question=item["question"],
        answer=item["answer"],
        sentences=listing,
    )
    try:
        raw = chat(_CONFIG["reader"], prompt)
        found = re.search(r"\{.*\}", raw, re.S)
        needed = json.loads(found.group(0))["needed"] if found else None
        row["needed"] = sorted({int(i) for i in needed if 0 <= int(i) < len(sentences)})
    except (*_CALL_ERRORS, AttributeError, TypeError, ValueError) as exc:
        row["needed"], row["error"] = None, f"{type(exc).__name__}: {exc}"[:200]
    return row


def _wiki(item: dict, root: Path) -> Engine:
    """The question's keep-everything wiki at ``root``, built unless already complete."""
    done = root / ".complete"
    if not done.exists():
        shutil.rmtree(root, ignore_errors=True)
    engine = Engine(root, KeepEverything(), embedder=_embedder())
    if done.exists():
        return engine
    for index, (date, session) in enumerate(zip(item["haystack_dates"], item["haystack_sessions"])):
        text = session_text(session)
        if text:
            metadata = {"role": "user", "date": iso_date(date)}
            engine.ingest(
                text, source_key=f"s{index:03d}", title=f"s{index:03d}", metadata=metadata
            )
    for claim in engine.store.claims(active_only=False):
        if claim["status"] != "active":
            engine.store.update_claim(claim["id"], {"status": "active"})
    done.write_text("")
    return engine


def _normal(text: str) -> str:
    return " ".join(text.split())


def score(task: tuple[dict, dict]) -> dict:
    item, labels = task
    with tempfile.TemporaryDirectory(prefix="jev-lme-sentences-") as scratch:
        cache = _CONFIG.get("wikis")
        root = Path(cache) / item["question_id"] if cache else Path(scratch)
        engine = _wiki(item, root)
        sid_of = {f"s{i:03d}": sid for i, sid in enumerate(item["haystack_session_ids"])}
        key_of = {s["id"]: s["source_key"] for s in engine.store.sources()}
        result = engine.recall(
            item["question"],
            limit=_CONFIG["limit"],
            max_chars=_CONFIG["max_chars"],
            offline=True,
            aggregate_limit=_CONFIG["aggregate_limit"],
            as_of=iso_date(item["question_date"]),
            neighbours=_CONFIG["neighbours"],
        )
    recalled = [(sid_of[key_of[i["source_id"]]], _normal(i["text"])) for i in result["items"]]
    chats = {sid for sid, _ in recalled}
    needed = []
    for index in labels.get("needed") or []:
        sentence = labels["sentences"][index]
        text = _normal(sentence["text"])
        # Sentence splitting may differ at a turn edge, so containment either way counts.
        found = any(
            sid == sentence["sid"] and (text in got or got in text) for sid, got in recalled
        )
        needed.append({**sentence, "found": found, "chat_found": sentence["sid"] in chats})
    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "needed": needed,
        "claims": len(result["items"]),
        "neighbours": result["neighbours"],
        "context_chars": len(result["context"]),
    }


def summarize(rows: list[dict]) -> dict:
    needed = [n for r in rows for n in r["needed"]]
    questions = [r for r in rows if r["needed"]]
    by_type = defaultdict(list)
    for row in questions:
        by_type[row["question_type"]].append(all(n["found"] for n in row["needed"]))
    return {
        "questions": len(questions),
        "answer_sentences": len(needed),
        "sentence_recall": round(sum(n["found"] for n in needed) / max(1, len(needed)), 3),
        "missed_in_recalled_chat": sum(n["chat_found"] and not n["found"] for n in needed),
        "missed_chat_not_recalled": sum(not n["chat_found"] for n in needed),
        "all_sentences_found": sum(all(n["found"] for n in r["needed"]) for r in questions),
        "all_found_by_type": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_type.items())},
        "mean_claims": round(sum(r["claims"] for r in rows) / max(1, len(rows)), 1),
        "mean_context_chars": round(sum(r["context_chars"] for r in rows) / max(1, len(rows))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="longmemeval_s_cleaned.json")
    parser.add_argument("--labels", required=True, help="answer-sentence labels (JSON lines)")
    parser.add_argument(
        "--label", action="store_true", help="first add labels for unlabelled questions"
    )
    parser.add_argument("--reader", default="gpt-oss:120b")
    parser.add_argument("--endpoint", default="https://ollama.com/api/chat")
    parser.add_argument("--wikis", help="keep each question's wiki here and reuse it")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--aggregate-limit", type=int)
    parser.add_argument("--neighbours", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output", help="write per-question rows here (JSON lines)")
    args = parser.parse_args()

    items = json.loads(Path(args.data).read_text())
    labels_path = Path(args.labels)
    config = {
        "reader": args.reader,
        "endpoint": args.endpoint,
        "wikis": args.wikis,
        "limit": args.limit,
        "max_chars": args.max_chars,
        "aggregate_limit": args.aggregate_limit,
        "neighbours": args.neighbours,
    }
    _CONFIG.update(config)
    labels = {}
    if labels_path.exists():
        labels = {r["question_id"]: r for r in map(json.loads, labels_path.open())}
    if args.label:
        todo = [i for i in items if labels.get(i["question_id"], {}).get("needed") is None]
        # Ollama Cloud caps concurrent requests: keep this low.
        with ThreadPoolExecutor(4) as pool, labels_path.open("a") as out:
            for row in pool.map(label, todo):
                labels[row["question_id"]] = row
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tasks = [(i, labels[i["question_id"]]) for i in items if i["question_id"] in labels]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(config,)) as pool:
        rows = list(pool.map(score, tasks, chunksize=4))
    print(json.dumps({"config": config, "summary": summarize(rows)}, indent=1))
    if args.output:
        Path(args.output).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
