# Aggregate recall for counting and date questions (2026-09-28)

The end-to-end QA run ([longmemeval-qa-2026-09-28.md](longmemeval-qa-2026-09-28.md))
lost most of its gap to the evidence ceiling on multi-session counting (0.71 against
0.88) and temporal arithmetic (0.82 against 0.95). Those questions need every mention
of something, and `recall` stopped at 20 claims.

## The change

`Engine.recall(..., aggregate_limit=N)` (CLI `recall --aggregate-limit N`, QA harness
`--aggregate-limit N`) raises the claim limit to N, at most 100, when the query looks
like it counts, totals, orders or dates events. `aggregation_query()` decides that
with one regex: "how many/much/long/often", "total", "combined", "number of",
"average", "first", "earliest", "latest", "before", "after", "since", "ago",
"between", "so far", "each", "every". No model call is made.

The extra claims come after the ranked shortlist, in the same fused order the
shortlist uses. They are never sent to JEV, so the rerank request keeps its existing
bounds (24 candidates, 14,000 bytes). A claim JEV saw and scored below the relevance
cut does not come back as an unranked extra. `max_chars` still bounds the context,
and no quote is truncated.

It is off by default (`aggregate_limit=None`), and the prompt hook doesn't use it.

## Results

All 500 LongMemEval_S questions, with the same setup as the first QA run: offline
keep-everything intake, embedder on, reader `gpt-oss:120b`, judge `glm-5.3`. Each
question was ingested once and recalled three ways. When two ways produced identical
notes, the answer was reused, so questions the regex doesn't match score the same in
every column.

| Type | n | 20 claims (baseline) | up to 40 | up to 60 | oracle (earlier run) |
|---|---:|---:|---:|---:|---:|
| multi-session | 121 | 0.719 | 0.777 | **0.818** | 0.884 |
| temporal-reasoning | 127 | 0.835 | **0.866** | 0.858 | 0.945 |
| knowledge-update | 72 | 0.917 | 0.931 | 0.903 | 0.944 |
| single-session-user | 64 | 0.953 | 0.953 | 0.953 | 1.000 |
| single-session-assistant | 56 | 0.321 | 0.304 | 0.321 | 0.411 |
| single-session-preference | 30 | 0.600 | 0.600 | 0.600 | 0.727 |
| abstention | 30 | 0.700 | 0.633 | 0.633 | 0.621 |
| **all** | 500 | 0.754 | 0.772 | **0.778** | 0.847 |

- The baseline reproduces the first run's 0.754 exactly.
- The regex matched 325 questions: 107 of 121 multi-session, 112 of 127 temporal,
  48 knowledge-update, 22 single-session-user, 12 assistant and 24 abstention. On
  those 325, accuracy went from 0.782 to 0.818 at 60.
- At 60, 21 answers went from wrong to right and 9 from right to wrong.
- Up to 60 closes 60% of the multi-session gap to the oracle and a quarter of the
  overall gap. Temporal gains 3 points and levels off past 40.
- Every evidence turn was recalled for 0.864 of questions at 20 claims, 0.900 at 40
  and 0.909 at 60. The median context grew from 2.9k to 8.1k characters.
- Abstention loses 2 of 30. With more notes, the reader answers from nearby mentions
  ("how long was I in Korea" gets a two-week trip) instead of saying it doesn't know.

The session-level proxy didn't predict this. Every labelled session was already in the
top 20 claims for 0.91 of multi-session and 0.89 of temporal questions. What was
missing was the other mentions within those sessions.

## Cost

- No TypeSafe calls. The QA run cost 1,500 reader and judge calls on Ollama Cloud.
- With 3,000 claims, extending to 60 or 100 claims adds under 20 ms to an offline
  recall (about 300 ms in total), well inside the prompt hook's 0.75 s budget. The
  hook still asks for 5 claims and 6,000 characters. That budget, not the limit,
  would cap it if the hook passed `aggregate_limit`.
- The JEV rerank request doesn't change size, because the extras are never reranked.

## Next

- Turn `aggregate_limit=60` on by default for CLI recall. It helps counting and date
  questions and leaves the rest unchanged, apart from the abstention cost above.
- Temporal still trails the oracle by 9 points, and more claims stopped helping past
  40. Those failures look like arithmetic over the right notes, not missing notes.

## Reproduce

```bash
export JEV_WIKI_EMBEDDING_MODEL=default OLLAMA_API_KEY=...
python packages/jev-wiki/examples/longmemeval_qa.py --data longmemeval_s_cleaned.json \
    --per-type 0 --workers 4 --aggregate-limit 60 --output wiki-agg60.json
```

Rows for all three variants (notes stats, answers, verdicts):
`/mnt/project-files/benchmarks/longmemeval-qa-aggregate-2026-09-28.jsonl` in the
project's shared folder.
