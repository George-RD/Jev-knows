"""Head-to-head on LongMemEval_S: jev-wiki against other ways of giving an agent memory.

Every system is scored the same way as ``longmemeval_qa.py``: it hands the reader notes,
the reader answers, and LongMemEval's own judge prompts grade the answer. Only the notes
differ, so the reader and judge are held fixed across systems.

Systems (all see only the user's turns, as the other harnesses do, so
``single-session-assistant`` questions are expected to fail everywhere):

- ``none``: no notes. The floor.
- ``full``: every chat, dated, in date order. At LongMemEval_S size the user turns
  are about 16k tokens, so they fit in the reader's context whole.
- ``bm25``: the five chats BM25 ranks highest for the question, dated. Plain RAG.
- ``jev``: jev-wiki with keep-everything intake and offline recall (BM25 plus the local
  embedder when ``JEV_WIKI_EMBEDDING_MODEL`` is set), aggregating up to 40 claims for
  counting and date questions, as the CLI does by default. No JEV calls.
- ``jev_live``: jev-wiki with real JEV intake at the shipped gates, then the same
  offline recall. Calls TypeSafe; the shared response cache answers the questions the
  live sweep already asked, and the report counts paid requests.
- ``llm_wiki``: Karpathy's LLM Wiki pattern. The builder model reads each chat in date
  order next to the current index (page titles and one-line summaries) and files the
  chat's facts, dated, onto new or existing pages. To answer, the model reads the index,
  picks up to ten pages, and the reader answers from those pages.
- ``llm_wiki_full``: the same wiki, every page handed to the reader (no page picking).
- ``oracle``: the labelled evidence chats only. The ceiling.

Wikis are cached per question and builder model in ``--wiki-cache``, so a rerun only pays
for reading and judging.

    python packages/jev-wiki/examples/memory_head_to_head.py \
        --data longmemeval_s_cleaned.json --per-type 10 --systems none,full,bm25,jev \
        --output /tmp/h2h.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

import longmemeval_qa as qa  # noqa: E402
from jev_wiki.engine import Engine  # noqa: E402
from jev_wiki.provider import JevProvider  # noqa: E402
from longmemeval import bm25_rank, default_cache_dir, select, session_text  # noqa: E402
from longmemeval_offline import _embedder  # noqa: E402

SYSTEMS = ("none", "full", "bm25", "jev", "jev_live", "llm_wiki", "llm_wiki_full", "oracle")
BM25_CHATS = 5
PAGE_PICKS = 10

WIKI_INGEST_PROMPT = (
    "You maintain a wiki about a user, built from your past chats with them. Each page has "
    "a title, a one-line summary and a list of dated facts. This is the current index:\n\n"
    "{index}\n\nA new chat, dated {date}. The user's messages:\n\n{text}\n\n"
    "File everything from this chat that could matter in a later conversation: facts about "
    "the user and their life, events and when they happened, preferences, plans, "
    "possessions, quantities, and anything that changes an earlier fact. Reuse an existing "
    "page title exactly where it fits; create a new page otherwise. Write each fact as one "
    "self-contained sentence about the user (no 'I'). Return JSON only, in this form:\n"
    '{{"pages": [{{"title": "...", "summary": "one line on what the page covers", '
    '"facts": ["..."]}}]}}\nReturn {{"pages": []}} if nothing is worth keeping.'
)

WIKI_PICK_PROMPT = (
    "This is the index of a wiki about a user, built from past chats with them. Each line "
    "is a page title, its summary and how many facts it holds.\n\n{index}\n\n"
    "Current date: {date}\nQuestion: {question}\n\nWhich pages should be read to answer the "
    "question? List at most {k} titles exactly as written, most useful first. Return JSON "
    'only: {{"pages": ["..."]}}'
)

_CONFIG: dict = {}


def call_json(model: str, prompt: str) -> dict:
    """One JSON-mode /api/chat call with low reasoning effort, retried like ``qa.chat``."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "think": "low",
        "options": {"temperature": 0},
    }
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"
    request = urllib.request.Request(
        _CONFIG["endpoint"], data=json.dumps(body).encode(), headers=headers
    )
    for attempt in range(qa.ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                content = json.load(response)["message"]["content"]
            match = re.search(r"\{.*\}", content, re.S)
            parsed = json.loads(match.group(0) if match else content)
            # A list or bare string is as useless as no answer; callers read .get().
            return parsed if isinstance(parsed, dict) else {}
        except qa._CALL_ERRORS as exc:
            status = getattr(exc, "code", None)
            if attempt == qa.ATTEMPTS - 1 or (
                status is not None and status < 500 and status != 429
            ):
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))
    raise AssertionError("unreachable")


def chats(item: dict) -> list[tuple[str, str]]:
    """(date, user text) of each non-empty chat, oldest first."""
    pairs = [
        (date, session_text(session))
        for date, session in zip(item["haystack_dates"], item["haystack_sessions"])
    ]
    return sorted((p for p in pairs if p[1]), key=lambda p: p[0])


def dated_blocks(pairs: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"### Chat on {date}\n{text}" for date, text in pairs)


def bm25_notes(item: dict) -> str:
    pairs = chats(item)
    top = bm25_rank(item["question"], [text for _, text in pairs])[:BM25_CHATS]
    return dated_blocks([pairs[i] for i in sorted(top)])


def _norm(title: str) -> str:
    return " ".join(title.lower().split())


def index_text(pages: dict) -> str:
    if not pages:
        return "(empty)"
    return "\n".join(
        f"- {p['title']}: {p['summary']} ({len(p['facts'])} facts)" for p in pages.values()
    )


def build_wiki(item: dict) -> dict:
    """Karpathy-style wiki for one question's haystack, cached on disk."""
    model = _CONFIG["wiki_model"]
    path = Path(_CONFIG["wiki_cache"]) / f"{item['question_id']}.{model.replace(':', '_')}.json"
    if path.exists():
        return json.loads(path.read_text())
    pages: dict[str, dict] = {}
    failures = 0
    for date, text in chats(item):
        prompt = WIKI_INGEST_PROMPT.format(index=index_text(pages), date=date, text=text)
        try:
            update = call_json(model, prompt)
        except qa._CALL_ERRORS:
            failures += 1
            continue
        for page in update.get("pages") or []:
            if not isinstance(page, dict) or not str(page.get("title", "")).strip():
                continue
            title = str(page["title"]).strip()
            entry = pages.setdefault(_norm(title), {"title": title, "summary": "", "facts": []})
            if page.get("summary"):
                entry["summary"] = str(page["summary"]).strip()
            facts = page.get("facts") or []
            if isinstance(facts, str):
                facts = [facts]  # Not one fact per character.
            entry["facts"] += [{"date": date, "text": str(f).strip()} for f in facts if f]
    wiki = {"model": model, "pages": pages, "failed_chats": failures}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(wiki))
    tmp.replace(path)
    return wiki


def page_text(page: dict) -> str:
    facts = "\n".join(f"- ({f['date']}) {f['text']}" for f in page["facts"])
    return f"## {page['title']}\n{facts}"


def llm_wiki_notes(item: dict, pick: bool) -> tuple[str, dict]:
    wiki = build_wiki(item)
    pages = wiki["pages"]
    stats = {
        "wiki_pages": len(pages),
        "wiki_facts": sum(len(p["facts"]) for p in pages.values()),
        "wiki_failed_chats": wiki["failed_chats"],
    }
    chosen = list(pages)
    if pick:
        answer = call_json(
            _CONFIG["wiki_model"],
            WIKI_PICK_PROMPT.format(
                index=index_text(pages),
                date=item["question_date"],
                question=item["question"],
                k=PAGE_PICKS,
            ),
        )
        names = [_norm(str(t)) for t in answer.get("pages") or []]
        chosen = [n for n in dict.fromkeys(names) if n in pages][:PAGE_PICKS]
        stats["wiki_pages_picked"] = len(chosen)
    return "\n\n".join(page_text(pages[n]) for n in chosen), stats


def jev_live_notes(item: dict) -> tuple[str, dict]:
    """Real JEV intake at the shipped gates, then the same offline recall as ``jev``."""
    provider = JevProvider(model=_CONFIG["jev_model"], cache_dir=_CONFIG["jev_cache"])
    date_of = {}
    with tempfile.TemporaryDirectory(prefix="jev-h2h-") as root:
        engine = Engine(root, provider, embedder=_embedder())
        for index, (date, session) in enumerate(
            zip(item["haystack_dates"], item["haystack_sessions"])
        ):
            key = f"s{index:03d}"
            date_of[key] = date
            text = session_text(session)
            if text:
                # Same role metadata as longmemeval.py, so its cached intake answers match.
                metadata = {"role": "user", "date": qa.iso_date(date)}
                result = engine.ingest(text, source_key=key, title=key, metadata=metadata)
                if result["status"] != "complete":
                    raise RuntimeError(f"ingestion incomplete: {result['status']}")
        key_of = {s["id"]: s["source_key"] for s in engine.store.sources()}
        stored = engine.store.claims(active_only=False)
        result = engine.recall(
            item["question"],
            limit=20,
            max_chars=20_000,
            offline=True,
            aggregate_limit=_CONFIG["aggregate_limit"],
            as_of=qa.iso_date(item["question_date"]),
        )
    notes = "\n".join(
        f"[{n}] (chat date: {date_of[key_of[i['source_id']]]}) "
        f"{json.dumps(i['text'], ensure_ascii=False)}"
        for n, i in enumerate(result["items"], 1)
    )
    telemetry = provider.telemetry
    return notes, {
        "claims": len(result["items"]),
        "claims_stored": len(stored),
        "claims_active": sum(c["status"] == "active" for c in stored),
        "jev_paid_requests": telemetry.get("cache_misses", 0),
        "jev_cache_hits": telemetry.get("cache_hits", 0),
    }


def notes_for(system: str, item: dict) -> tuple[str, dict]:
    if system == "none":
        return "", {}
    if system == "full":
        return dated_blocks(chats(item)), {}
    if system == "bm25":
        return bm25_notes(item), {}
    if system == "oracle":
        return qa.oracle_notes(item), {}
    if system == "jev":
        return qa.wiki_notes(item)
    if system == "jev_live":
        return jev_live_notes(item)
    if system == "llm_wiki":
        return llm_wiki_notes(item, pick=True)
    if system == "llm_wiki_full":
        return llm_wiki_notes(item, pick=False)
    raise ValueError(system)


def run(task: tuple[str, dict]) -> dict:
    system, item = task
    row = {
        "system": system,
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "abstention": item["question_id"].endswith("_abs"),
        "question": item["question"],
        "answer": str(item["answer"]),
    }
    started = time.perf_counter()
    try:
        notes, stats = notes_for(system, item)
    except qa._CALL_ERRORS + (RuntimeError,) as exc:
        row["error"] = f"notes: {type(exc).__name__}: {exc}"[:300]
        return row
    row.update(stats)
    row["notes_seconds"] = round(time.perf_counter() - started, 2)
    row["notes_chars"] = len(notes)
    prompt = qa.READER_PROMPT.format(
        notes=notes or "(none)", date=item["question_date"], question=item["question"]
    )
    try:
        row["hypothesis"] = qa.chat(_CONFIG["reader"], prompt)
        verdict = qa.chat(
            _CONFIG["judge"], qa.judge_prompt({**item, "answer": row["answer"]}, row["hypothesis"])
        )
        row["judge_raw"] = verdict[:200]
        row["correct"] = "yes" in verdict.lower()
    except qa._CALL_ERRORS as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


def _init(config: dict) -> None:
    _CONFIG.update(config)
    qa._CONFIG.update(config)


def summarize(rows: list[dict], systems: list[str]) -> dict:
    by_system = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)

    def accuracy(subset):
        graded = [r["correct"] for r in subset if "correct" in r]
        return round(statistics.mean(graded), 3) if graded else None

    out = {}
    for system in systems:
        subset = by_system[system]
        types = defaultdict(list)
        for r in subset:
            types["abstention" if r["abstention"] else r["question_type"]].append(r)
        out[system] = {
            "questions": len(subset),
            "errors": sum("error" in r for r in subset),
            "accuracy": accuracy(subset),
            "accuracy_without_assistant_questions": accuracy(
                [r for r in subset if r["question_type"] != "single-session-assistant"]
            ),
            "mean_notes_chars": round(statistics.mean(r.get("notes_chars", 0) for r in subset))
            if subset
            else 0,
            "by_type": {k: accuracy(v) for k, v in sorted(types.items())},
        }
        if system == "jev_live":
            out[system]["jev_paid_requests"] = sum(r.get("jev_paid_requests", 0) for r in subset)
    # Paired against jev: questions one system gets right and the other wrong.
    if "jev" in by_system:
        jev = {r["question_id"]: r.get("correct") for r in by_system["jev"]}
        for system in systems:
            if system == "jev":
                continue
            pairs = [
                (jev[r["question_id"]], r["correct"])
                for r in by_system[system]
                if "correct" in r and jev.get(r["question_id"]) is not None
            ]
            out[system]["vs_jev"] = {
                "paired": len(pairs),
                "only_this_right": sum(b and not a for a, b in pairs),
                "only_jev_right": sum(a and not b for a, b in pairs),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="longmemeval_s_cleaned.json")
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument(
        "--no-abstention",
        action="store_true",
        help="drop _abs questions; with --per-type 5 this is the live sweep's cached set",
    )
    parser.add_argument("--reader", default="gpt-oss:120b")
    parser.add_argument("--judge", default="glm-5.3")
    parser.add_argument(
        "--wiki-model", default="gpt-oss:120b", help="builds and reads the LLM wiki"
    )
    parser.add_argument("--endpoint", default="https://ollama.com/api/chat")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--aggregate-limit", type=int, default=40)
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--jev-cache", default=str(default_cache_dir()))
    parser.add_argument(
        "--wiki-cache", default=str(Path.home() / ".cache" / "jev-wiki" / "llm-wiki")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true", help="keep graded rows in --output")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        parser.error(f"unknown systems: {sorted(unknown)}")
    items = select(
        json.loads(Path(args.data).read_text()),
        args.per_type,
        args.seed,
        include_abstention=not args.no_abstention,
    )
    config = {
        "endpoint": args.endpoint,
        "reader": args.reader,
        "judge": args.judge,
        "wiki_model": args.wiki_model,
        "wiki_cache": args.wiki_cache,
        "aggregate_limit": args.aggregate_limit,
        "jev_model": args.jev_model,
        "jev_cache": args.jev_cache,
    }
    done = {}
    output = Path(args.output)
    if args.resume and output.exists():
        for row in json.loads(output.read_text())["rows"]:
            if "correct" in row:
                done[(row["system"], row["question_id"])] = row
    todo = [
        (system, item)
        for system in systems
        for item in items
        if (system, item["question_id"]) not in done
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(config,)) as pool:
        for count, row in enumerate(pool.map(run, todo, chunksize=1), 1):
            done[(row["system"], row["question_id"])] = row
            if count % 10 == 0:
                print(f"{count}/{len(todo)} rows", file=sys.stderr, flush=True)
    rows = [
        done[(s, i["question_id"])] for s in systems for i in items if (s, i["question_id"]) in done
    ]
    report = {
        "config": {
            **config,
            "per_type": args.per_type,
            "seed": args.seed,
            "abstention": not args.no_abstention,
            "systems": systems,
            "embedding_model": os.environ.get("JEV_WIKI_EMBEDDING_MODEL"),
        },
        "seconds": round(time.perf_counter() - started, 1),
        "summary": summarize(rows, systems),
    }
    print(json.dumps(report["summary"], indent=1))
    output.write_text(json.dumps({**report, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
