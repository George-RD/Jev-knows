"""Head-to-head on MultiHop-RAG: questions whose answer connects facts across articles.

LongMemEval (``memory_head_to_head.py``) mostly asks for one fact from one chat. An LLM
Wiki is built for something else: filing facts from many sources under shared pages so
that questions which link them can be answered. MultiHop-RAG (Tang and Yang, COLM 2024)
tests that. Its questions each need 2 to 4 news articles: who links two reports
(inference), whether two sources agree (comparison), how coverage changed over time
(temporal), and questions the corpus cannot answer (null).

One corpus is shared by every question, as in a real knowledge base: the evidence
articles of the chosen questions plus seeded distractors. Every system builds its
memory once from that corpus, and one reader and judge grade them all, with each
question answered ``--samples`` times and graded by majority.

Systems:

- ``none``: no notes.
- ``bm25``: the 20 best BM25 chunks (about 1,000 characters each), the plain RAG baseline.
- ``jev``: jev-wiki with keep-everything intake and offline recall at the CLI's
  defaults (up to 40 claims for aggregate questions, neighbouring claims, date windows).
- ``jev_wide``: the same, but every question recalls up to 40 claims as an aggregate
  question would, so jev-wiki's notes are about as long as BM25's.
- ``jev_passages``: the same recall, but each recalled claim is returned as the passage
  around it in its article (paragraphs out to about ``PASSAGE_CHARS``), overlapping
  passages merged, up to ``NOTES_CHARS``: claims find the evidence, passages carry it.
- ``llm_wiki``: Karpathy's LLM Wiki. The builder model reads each article in date order
  next to the current index and files its facts on new or existing pages; to answer, the
  model picks up to ten pages from the index.
- ``llm_wiki_search``: the same wiki searched instead of browsed, as Karpathy suggests
  once a wiki outgrows its index: whole pages in BM25 order, up to ``NOTES_CHARS``.
- ``llm_wiki_full``: every page of that wiki, when it fits ``--max-wiki-chars``.
- ``oracle``: the question's evidence articles.

Every article is shown to every system as "source, date: title" followed by its body.
The wiki is cached in ``--wiki-cache`` and resumes after an interruption.

    python packages/jev-wiki/examples/multihop_head_to_head.py \
        --questions MultiHopRAG.json --corpus corpus.json --per-type 20 --output /tmp/mh.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import longmemeval_qa as qa  # noqa: E402
import memory_head_to_head as h2h  # noqa: E402
import jev_wiki.engine as engine_module  # noqa: E402
from jev_wiki.engine import Engine  # noqa: E402
from longmemeval import bm25_rank  # noqa: E402
from longmemeval_offline import KeepEverything, _embedder  # noqa: E402

SYSTEMS = (
    "none",
    "bm25",
    "jev",
    "jev_wide",
    "jev_passages",
    "llm_wiki",
    "llm_wiki_search",
    "llm_wiki_full",
    "oracle",
)
BM25_CHUNKS = 20
NOTES_CHARS = 20_000
PASSAGE_CHARS = 1000
CHUNK_CHARS = 1000

READER_PROMPT = (
    "Below are notes drawn from a collection of news articles. Answer the question using "
    "only these notes. Answer briefly: a name, yes or no, or a short phrase. If the notes "
    "do not contain enough to answer, reply 'Insufficient information.'\n\n"
    "Notes:\n\n{notes}\n\nQuestion: {question}\nAnswer:"
)

WIKI_INGEST_PROMPT = (
    "You maintain a wiki built from news articles. Each page covers one entity, event or "
    "topic and holds a one-line summary and a list of facts. This is the current index:"
    "\n\n{index}\n\nA new article:\n\n{text}\n\nFile every fact from this article that a "
    "later question could need: who did what, when, numbers, claims, opinions, and how "
    "this relates to people, companies and events already in the wiki. Put each fact on "
    "every page it concerns. Reuse an existing page title exactly where it fits; create a "
    "new page otherwise. Write each fact as one self-contained sentence that names the "
    "source and date of the report. Return JSON only, in this form:\n"
    '{{"pages": [{{"title": "...", "summary": "one line on what the page covers", '
    '"facts": ["..."]}}]}}'
)

WIKI_PICK_PROMPT = (
    "This is the index of a wiki built from news articles. Each line is a page title, its "
    "summary and how many facts it holds.\n\n{index}\n\nQuestion: {question}\n\nWhich "
    "pages should be read to answer the question? List at most {k} titles exactly as "
    'written, most useful first. Return JSON only: {{"pages": ["..."]}}'
)

_CONFIG: dict = {}
_STATE: dict = {}
_AGGREGATION_QUERY = engine_module.aggregation_query


def header(article: dict) -> str:
    return f"{article['source']}, {article['published_at'][:10]}: {article['title']}"


def article_text(article: dict) -> str:
    return f"{header(article)}\n\n{article['body'].strip()}"


def select(questions: list[dict], per_type: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for q in questions:
        groups[q["question_type"]].append(q)
    chosen = []
    for kind in sorted(groups):
        chosen += rng.sample(groups[kind], min(per_type, len(groups[kind])))
    return chosen


def build_corpus(corpus: list[dict], questions: list[dict], size: int, seed: int) -> list[dict]:
    """The chosen questions' evidence articles plus seeded distractors, oldest first."""
    evidence = {e["title"] for q in questions for e in q["evidence_list"]}
    picked = [a for a in corpus if a["title"] in evidence]
    rest = [a for a in corpus if a["title"] not in evidence]
    picked += random.Random(seed).sample(rest, max(0, min(len(rest), size - len(picked))))
    return sorted(picked, key=lambda a: (a["published_at"], a["title"]))


def chunks(text: str) -> list[str]:
    """Paragraph-aligned pieces of about CHUNK_CHARS characters."""
    out, current = [], ""
    for paragraph in text.split("\n"):
        if current and len(current) + len(paragraph) > CHUNK_CHARS:
            out.append(current.strip())
            current = ""
        current += paragraph + "\n"
    if current.strip():
        out.append(current.strip())
    return out


def build_jev(articles: list[dict], root: str) -> None:
    engine = Engine(root, KeepEverything(), embedder=_embedder())
    for index, article in enumerate(articles):
        engine.ingest(
            article["body"].strip(),
            source_key=f"a{index:04d}",
            title=header(article),
            metadata={"role": "document", "date": article["published_at"][:10]},
        )
    for claim in engine.store.claims(active_only=False):
        if claim["status"] != "active":
            engine.store.update_claim(claim["id"], {"status": "active"})


def build_llm_wiki(articles: list[dict], path: Path) -> dict:
    """Sequential Karpathy-style build over the corpus, checkpointed after each article."""
    state = {"model": _CONFIG["wiki_model"], "done": 0, "failed": 0, "pages": {}}
    if path.exists():
        state = json.loads(path.read_text())
    pages = state["pages"]
    for article in articles[state["done"] :]:
        prompt = WIKI_INGEST_PROMPT.format(index=h2h.index_text(pages), text=article_text(article))
        try:
            update = h2h.call_json(_CONFIG["wiki_model"], prompt)
        except qa._CALL_ERRORS:
            update = {}
            state["failed"] += 1
        date = article["published_at"][:10]
        for page in update.get("pages") or []:
            if not isinstance(page, dict) or not str(page.get("title", "")).strip():
                continue
            title = str(page["title"]).strip()
            entry = pages.setdefault(h2h._norm(title), {"title": title, "summary": "", "facts": []})
            if page.get("summary"):
                entry["summary"] = str(page["summary"]).strip()
            facts = page.get("facts") or []
            if isinstance(facts, str):
                facts = [facts]
            entry["facts"] += [{"date": date, "text": str(f).strip()} for f in facts if f]
        state["done"] += 1
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(path)
        if state["done"] % 10 == 0:
            print(f"wiki: {state['done']}/{len(articles)} articles", file=sys.stderr, flush=True)
    return state


def _init(config: dict, state: dict) -> None:
    _CONFIG.update(config)
    h2h._CONFIG.update(config)
    qa._CONFIG.update(config)
    _STATE.update(state)


def jev_recall(question: dict, wide: bool = False) -> dict:
    if "engine" not in _STATE:
        _STATE["engine"] = Engine(_STATE["jev_root"], KeepEverything(), embedder=_embedder())
        _STATE["title_of"] = {s["id"]: s["title"] for s in _STATE["engine"].store.sources()}
    # jev_wide treats every question as an aggregate one; the engine reads this global.
    engine_module.aggregation_query = (lambda _query: True) if wide else _AGGREGATION_QUERY
    return _STATE["engine"].recall(
        question["query"],
        limit=20,
        max_chars=20_000,
        offline=True,
        aggregate_limit=_CONFIG["aggregate_limit"],
        neighbours=True,
        window_claims=True,
    )


def jev_notes(question: dict, wide: bool = False) -> tuple[str, dict]:
    result = jev_recall(question, wide)
    notes = "\n".join(
        f"[{n}] ({_STATE['title_of'][i['source_id']]}) {json.dumps(i['text'], ensure_ascii=False)}"
        for n, i in enumerate(result["items"], 1)
    )
    return notes, {"claims": len(result["items"])}


def passage(text: str, start: int, end: int) -> tuple[int, int]:
    """Whole paragraphs around ``text[start:end]``, widened to about PASSAGE_CHARS."""
    left = text.rfind("\n", 0, start) + 1
    right = text.find("\n", end)
    right = len(text) if right == -1 else right
    while right - left < PASSAGE_CHARS and (left > 0 or right < len(text)):
        if left > 0:
            left = text.rfind("\n", 0, left - 1) + 1
        if right < len(text) and right - left < PASSAGE_CHARS:
            nxt = text.find("\n", right + 1)
            right = len(text) if nxt == -1 else nxt
    return left, right


def jev_passage_notes(question: dict) -> tuple[str, dict]:
    """jev-wiki recall, each claim widened to its passage; overlapping passages merge."""
    result = jev_recall(question)
    engine = _STATE["engine"]
    if "claim_span" not in _STATE:
        _STATE["claim_span"] = {
            c["id"]: (c["source_id"], c["start"], c["end"]) for c in engine.store.claims()
        }
        _STATE["source_text"] = {}
    spans: dict[str, list[list[int]]] = {}
    order: list[str] = []
    size = 0
    for item in result["items"]:
        source_id, start, end = _STATE["claim_span"][item["id"]]
        texts = _STATE["source_text"]
        text = texts.setdefault(source_id, engine.store.read_source(source_id))
        left, right = passage(text, start, end)
        ranges = spans.setdefault(source_id, [])
        if source_id not in order:
            order.append(source_id)
        overlap = [r for r in ranges if r[0] <= right and left <= r[1]]
        new_left = min([left] + [r[0] for r in overlap])
        new_right = max([right] + [r[1] for r in overlap])
        added = (new_right - new_left) - sum(r[1] - r[0] for r in overlap)
        if size + added > NOTES_CHARS:
            continue
        for r in overlap:
            ranges.remove(r)
        ranges.append([new_left, new_right])
        size += added
    blocks = []
    for source_id in order:
        text = _STATE["source_text"][source_id]
        body = "\n[...]\n".join(text[a:b].strip() for a, b in sorted(spans[source_id]))
        blocks.append(f"### {_STATE['title_of'][source_id]}\n{body}")
    return "\n\n".join(blocks), {"claims": len(result["items"]), "sources": len(order)}


def notes_for(system: str, question: dict) -> tuple[str, dict]:
    if system == "none":
        return "", {}
    if system == "bm25":
        pieces = _STATE["chunks"]
        top = bm25_rank(question["query"], [p for _, p in pieces])[:BM25_CHUNKS]
        return "\n\n".join(f"### {pieces[i][0]}\n{pieces[i][1]}" for i in top), {}
    if system == "oracle":
        titles = {e["title"] for e in question["evidence_list"]}
        return "\n\n".join(article_text(a) for a in _STATE["articles"] if a["title"] in titles), {}
    if system == "jev_passages":
        return jev_passage_notes(question)
    if system in ("jev", "jev_wide"):
        return jev_notes(question, wide=system == "jev_wide")
    pages = _STATE["wiki"]["pages"]
    if system == "llm_wiki_search":
        texts = _STATE.setdefault("page_texts", [h2h.page_text(p) for p in pages.values()])
        picked, size = [], 0
        for i in bm25_rank(question["query"], texts):
            if size + len(texts[i]) > NOTES_CHARS:
                continue  # A page too big for the room left; smaller ones may still fit.
            picked.append(texts[i])
            size += len(texts[i]) + 2
        return "\n\n".join(picked), {"pages_picked": len(picked)}
    if system == "llm_wiki_full":
        return "\n\n".join(h2h.page_text(p) for p in pages.values()), {}
    if system == "llm_wiki":
        answer = h2h.call_json(
            _CONFIG["wiki_model"],
            WIKI_PICK_PROMPT.format(
                index=h2h.index_text(pages), question=question["query"], k=h2h.PAGE_PICKS
            ),
        )
        names = [h2h._norm(str(t)) for t in answer.get("pages") or []]
        chosen = [n for n in dict.fromkeys(names) if n in pages][: h2h.PAGE_PICKS]
        return "\n\n".join(h2h.page_text(pages[n]) for n in chosen), {"pages_picked": len(chosen)}
    raise ValueError(system)


def grade(question: dict, notes: str) -> dict:
    """Majority of ``samples`` answers from the same notes, as ``qa.answer`` does."""
    prompt = READER_PROMPT.format(notes=notes or "(none)", question=question["query"])
    kind = "abstention" if question["question_type"] == "null_query" else "default"
    votes, first = [], {}
    try:
        for _ in range(_CONFIG["samples"]):
            hypothesis = qa.chat(_CONFIG["reader"], prompt)
            verdict = qa.chat(
                _CONFIG["judge"],
                qa.JUDGE_PROMPTS[kind].format(
                    question=question["query"], answer=question["answer"], response=hypothesis
                ),
            )
            votes.append("yes" in verdict.lower())
            first = first or {"hypothesis": hypothesis, "judge_raw": verdict[:200]}
    except qa._CALL_ERRORS as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return {**first, "votes": sum(votes), "correct": 2 * sum(votes) > len(votes)}


def run(task: tuple[str, int, dict]) -> dict:
    system, index, question = task
    row = {
        "system": system,
        "index": index,
        "question_type": question["question_type"],
        "query": question["query"],
        "answer": question["answer"],
    }
    try:
        notes, stats = notes_for(system, question)
    except qa._CALL_ERRORS as exc:
        return {**row, "error": f"notes: {type(exc).__name__}: {exc}"[:300]}
    row.update(stats, notes_chars=len(notes))
    row.update(grade(question, notes))
    return row


def summarize(rows: list[dict], systems: list[str]) -> dict:
    out = {}
    by = defaultdict(list)
    for row in rows:
        by[row["system"]].append(row)
    jev = {r["index"]: r.get("correct") for r in by["jev"]}
    for system in systems:
        subset = by[system]
        graded = [r for r in subset if "correct" in r]
        types = defaultdict(list)
        for r in graded:
            types[r["question_type"]].append(r["correct"])
        out[system] = {
            "questions": len(subset),
            "errors": len(subset) - len(graded),
            "accuracy": round(statistics.mean(r["correct"] for r in graded), 3) if graded else None,
            "by_type": {k: round(statistics.mean(v), 3) for k, v in sorted(types.items())},
            "mean_notes_chars": round(statistics.mean(r.get("notes_chars", 0) for r in subset))
            if subset
            else 0,
        }
        if system != "jev" and jev:
            pairs = [
                (jev[r["index"]], r["correct"]) for r in graded if jev.get(r["index"]) is not None
            ]
            out[system]["vs_jev"] = {
                "only_this_right": sum(b and not a for a, b in pairs),
                "only_jev_right": sum(a and not b for a, b in pairs),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", required=True, help="MultiHopRAG.json")
    parser.add_argument("--corpus", required=True, help="corpus.json")
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--corpus-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--reader", default="gpt-oss:120b")
    parser.add_argument("--judge", default="glm-5.3")
    parser.add_argument("--wiki-model", default="gpt-oss:120b")
    parser.add_argument("--endpoint", default="https://ollama.com/api/chat")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--aggregate-limit", type=int, default=40)
    parser.add_argument("--max-wiki-chars", type=int, default=300_000)
    parser.add_argument(
        "--wiki-cache", default=str(Path.home() / ".cache" / "jev-wiki" / "multihop-wiki")
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true", help="keep graded rows in --output")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if set(systems) - set(SYSTEMS):
        parser.error(f"unknown systems: {sorted(set(systems) - set(SYSTEMS))}")
    questions = select(json.loads(Path(args.questions).read_text()), args.per_type, args.seed)
    articles = build_corpus(
        json.loads(Path(args.corpus).read_text()), questions, args.corpus_size, args.seed
    )
    config = {
        "endpoint": args.endpoint,
        "reader": args.reader,
        "judge": args.judge,
        "wiki_model": args.wiki_model,
        "samples": args.samples,
        "aggregate_limit": args.aggregate_limit,
    }
    _init(config, {})
    state: dict = {
        "articles": articles,
        "chunks": [(header(a), c) for a in articles for c in chunks(a["body"])],
    }
    report_extra = {
        "corpus_articles": len(articles),
        "corpus_chars": sum(len(a["body"]) for a in articles),
    }
    if {"llm_wiki", "llm_wiki_search", "llm_wiki_full"} & set(systems):
        name = f"seed{args.seed}-q{args.per_type}-c{args.corpus_size}.{args.wiki_model.replace(':', '_')}.json"
        cache = Path(args.wiki_cache)
        cache.mkdir(parents=True, exist_ok=True)
        wiki = build_llm_wiki(articles, cache / name)
        state["wiki"] = wiki
        full_chars = sum(len(h2h.page_text(p)) + 2 for p in wiki["pages"].values())
        report_extra.update(
            wiki_pages=len(wiki["pages"]),
            wiki_facts=sum(len(p["facts"]) for p in wiki["pages"].values()),
            wiki_failed_articles=wiki["failed"],
            wiki_full_chars=full_chars,
        )
        if "llm_wiki_full" in systems and full_chars > args.max_wiki_chars:
            print(f"llm_wiki_full skipped: {full_chars} chars", file=sys.stderr)
            systems.remove("llm_wiki_full")
            report_extra["llm_wiki_full_skipped"] = True
    with tempfile.TemporaryDirectory(prefix="jev-multihop-") as root:
        if {"jev", "jev_wide", "jev_passages"} & set(systems):
            started = time.perf_counter()
            build_jev(articles, root)
            report_extra["jev_build_seconds"] = round(time.perf_counter() - started, 1)
        state["jev_root"] = root
        done = {}
        output = Path(args.output)
        if args.resume and output.exists():
            for row in json.loads(output.read_text())["rows"]:
                if "correct" in row:
                    done[(row["system"], row["index"])] = row
        todo = [(s, i, q) for s in systems for i, q in enumerate(questions) if (s, i) not in done]
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(config, state)) as pool:
            for count, row in enumerate(pool.map(run, todo, chunksize=1), 1):
                done[(row["system"], row["index"])] = row
                if count % 10 == 0:
                    print(f"{count}/{len(todo)} rows", file=sys.stderr, flush=True)
    rows = [done[(s, i)] for s in systems for i in range(len(questions)) if (s, i) in done]
    report = {
        "benchmark": "MultiHop-RAG",
        "config": {
            **config,
            "per_type": args.per_type,
            "corpus_size": args.corpus_size,
            "seed": args.seed,
            "systems": systems,
            "embedding_model": os.environ.get("JEV_WIKI_EMBEDDING_MODEL"),
        },
        **report_extra,
        "summary": summarize(rows, systems),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=1))
    output.write_text(json.dumps({**report, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
