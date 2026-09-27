# LongMemEval_S retrieval — 27 September 2026

Harness: `examples/longmemeval.py`. Benchmark: LongMemEval_S (cleaned), about 48 chat
sessions per question, scored as session-level recall_any@k. Model `jev-1.13.0`, cache
off, 12 workers, all 470 questions without the `_abs` suffix (seed 0).

## Run validity

The TypeSafe account ran out of credits after 351 questions (the API now returns
`billing_error`). The remaining 119 questions ingested with most sources deferred and
are excluded, together with 5 questions whose JEV rerank fell back to lexical order.
**346 questions are valid**: every type except temporal reasoning is complete
(temporal: 3 of 133). Those 346 questions used 68M input tokens (about $2.90 at list price) in 35k requests.

This run predates two harness fixes merged in #3: deferred ingestion now fails the
question instead of being scored, and answer-turn retention is measured by claim
offsets. The first is applied here by filtering. The second affects only the
diagnostic answer-turn column, so that column is omitted.

## All valid questions (n=346)

BM25 over each session's raw user turns (no retention step) reaches recall@1
0.85, @5 0.95, @10 0.96.

| Policy | Active claims per question | Evidence sessions with an active claim | wiki lexical @10 | wiki JEV @10 | BM25 claims @10 | BM25 claims + JEV @1 |
|---|---|---|---|---|---|---|
| shipped | 0.1 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 |
| keep>=0.6 | 1.7 | 0.07 | 0.10 | 0.02 | 0.10 | 0.02 |
| keep>=0.4 | 7.9 | 0.25 | 0.32 | 0.12 | 0.32 | 0.12 |
| keep>=0.2 | 21.7 | 0.51 | 0.52 | 0.22 | 0.53 | 0.22 |
| keep_top_choice | 29.9 | 0.59 | 0.59 | 0.28 | 0.60 | 0.27 |
| all_stored | 207.6 | 0.98 | 0.88 | 0.76 | 0.93 | 0.80 |

Answer stated in a user turn (n=295; BM25 reference @10 0.97):

| Policy | Active claims per question | Evidence sessions with an active claim | wiki lexical @10 | wiki JEV @10 | BM25 claims @10 | BM25 claims + JEV @1 |
|---|---|---|---|---|---|---|
| shipped | 0.1 | 0.01 | 0.01 | 0.00 | 0.01 | 0.00 |
| keep>=0.6 | 1.7 | 0.08 | 0.11 | 0.02 | 0.11 | 0.02 |
| keep>=0.4 | 7.9 | 0.27 | 0.36 | 0.14 | 0.36 | 0.14 |
| keep>=0.2 | 21.6 | 0.53 | 0.59 | 0.25 | 0.59 | 0.25 |
| keep_top_choice | 29.7 | 0.63 | 0.66 | 0.32 | 0.68 | 0.31 |
| all_stored | 206.1 | 1.00 | 0.94 | 0.86 | 0.97 | 0.88 |

## By question type (recall@10)

| Type | n | BM25 sessions | wiki lexical, shipped | wiki lexical, keep top choice | wiki lexical, all stored | BM25 claims, all stored |
|---|---|---|---|---|---|---|
| knowledge-update | 72 | 1.00 | 0.01 | 0.75 | 1.00 | 1.00 |
| multi-session | 121 | 0.97 | 0.02 | 0.76 | 0.98 | 0.97 |
| single-session-assistant | 56 | 0.95 | 0.00 | 0.14 | 0.54 | 0.75 |
| single-session-preference | 30 | 0.80 | 0.00 | 0.30 | 0.57 | 0.83 |
| single-session-user | 64 | 1.00 | 0.02 | 0.58 | 0.97 | 0.98 |
| temporal-reasoning | 3 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |

## Findings

1. **The shipped gate retains almost nothing.** On average 0.12 claims per question
   become active, out of about 208 stored, and recall@10 is 0.01.
2. **Lowering the threshold is not enough.** Even when a claim is accepted whenever
   JEV's top keep choice is `keep`, only 59% of evidence sessions have any active
   claim and recall@10 is 0.59. JEV labels most personal facts in chatty turns
   `discard` or `review`, for example "I've completed three courses on Coursera,
   and I'm excited to dive deeper into CNNs…".
3. **Retrieval is sound once evidence is retained.** With every stored claim active,
   BM25 over claims reaches 0.93 against 0.97 for raw sessions. The engine's
   overlap-count shortlist reaches 0.88.
4. **JEV reranking trades recall for precision.** It keeps only candidates scored at
   least "useful supporting context", so recall@1 equals recall@10. With all claims
   active it scores 0.80 at @1 over a BM25 shortlist, against 0.85 for plain BM25 over sessions.
5. **Assistant-turn answers are out of reach by design.** Single-session-assistant
   questions only succeed through user-turn context, because assistant claims never
   activate.

## Recommendation

Keep the 0.82 default until intake changes. Split candidates into sentences with
surrounding context, and reword the keep question so stated personal facts count as
durable, then rerun this sweep. Replace the overlap-count shortlist with BM25 or
another IDF-weighted ranking, and consider accepting reranked candidates at a lower
relevance cut-off when recall matters.
