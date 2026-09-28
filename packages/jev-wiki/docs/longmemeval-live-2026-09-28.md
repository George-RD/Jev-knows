# Live LongMemEval sweep on main — 28 September 2026

The 30-question LongMemEval_S sweep (`examples/longmemeval.py --per-type 5 --seed 0
--workers 10`, model `jev-1.13.0`, no abstention items) rerun live on main after the
BM25 shortlist (PR #6), source context (PR #7), embedding candidates (PRs #8/#9, off
here because `JEV_WIKI_EMBEDDING_MODEL` is unset) and the response cache (PR #10).
Intake is unchanged since the `wiki-v2` run in
[intake-sentence-claims-2026-09-27.md](intake-sentence-claims-2026-09-27.md), which is
the baseline. This run adds keep thresholds 0.75, 0.7, 0.65 and 0.5 to the harness so
the threshold decision has finer data. The default keep threshold stays at 0.82.

No failures, no retries, no provider validation losses (the baseline lost 14 of 360
reranks). Wall time 12.5 minutes.

## Recall at the shipped gate (recall_any@10 over sessions)

| | Baseline (27 Sep, `wiki-v2`) | This run |
| --- | --- | --- |
| Lexical recall (`wiki_lexical`) | 0.63 | **0.67** |
| JEV rerank (`wiki_jev`) | 0.60 | 0.63 |
| BM25 over raw user turns | 0.93 | 0.93 |
| Candidates activated | 13.4% | 13.3% |
| Labelled answer turns overlapped by an active claim | 71% | 69% |

The gain is one question (single-session-preference, 0.0 to 0.2), in line with the
offline source-context result. Everything else at the shipped gate is as before:
knowledge-update, multi-session and single-session-user 1.0, temporal reasoning 0.8,
single-session-assistant 0.0 (answers are in assistant turns, which are never
ingested).

## Keep-threshold sweep

Each row activates every stored claim whose top keep choice is `keep` at or above the
threshold and whose kind is anything but `uncertain` with confidence at or above
min(threshold, 0.70). 20,502
candidates, 15,929 stored claims over 30 questions.

| Keep threshold | Candidates active | Evidence sessions retained | Answer turns active | Lexical recall@10 | JEV rerank recall@10 |
| --- | --- | --- | --- | --- | --- |
| 0.82 (shipped) | 13.3% | 0.83 | 0.69 | 0.67 | 0.63 |
| 0.75 | 14.7% | 0.85 | 0.69 | 0.67 | 0.63 |
| 0.7 | 15.5% | 0.85 | 0.69 | 0.67 | 0.63 |
| 0.65 | 17.3% | 0.87 | 0.76 | **0.73** | 0.67 |
| 0.6 | 19.1% | 0.89 | 0.78 | 0.73 | 0.67 |
| 0.5 | 23.1% | 0.92 | 0.84 | 0.73 | 0.67 |
| 0.4 | 27.5% | 0.92 | 0.87 | 0.70 | 0.67 |
| 0.2 | 33.5% | 0.94 | 0.93 | 0.73 | 0.73 |
| keep top choice | 34.1% | 0.94 | 0.93 | 0.73 | 0.73 |
| every stored claim | 77.7% | 0.98 | 1.00 | 0.87 | 0.80 |

Reading it for the threshold decision:

- 0.82 down to 0.7 changes nothing measurable in recall; it only adds 2 points of
  active claims.
- 0.65 is the first step that gains recall (0.67 to 0.73, two questions) at 30% more
  active claims than shipped. Below 0.65, recall stays at 0.70–0.73 while the store keeps
  growing (0.5 activates 1.7 times as many claims, 0.2 about 2.5 times).
- On 30 questions one question is 0.033, so the whole 0.65–top-choice band is within
  two questions of itself. The step at 0.65 is the only consistent signal.

## Finding: the JEV rerank keeps about one claim

At every policy the JEV rerank's recall@1 equals its recall@10, and at the shipped gate
it returns 1.1 claims per question on average: the `relevance >= 1.5` cut in the
rerank drops almost everything after the top hit. This is why `wiki_jev` sits at or
below lexical order at every threshold. The rerank cut, not intake, is the next place
recall is lost once intake keeps the evidence.

## Cost and cache

| | This run |
| --- | --- |
| Paid requests | 3,716 |
| Input / output tokens | 14.03M / 3.65M |
| Cache hits (identical rerank asks across policies) | 116 (360K input tokens saved) |

The response cache was written to the shared project folder,
`/mnt/project-files/jev-cache/longmemeval` (3,716 entries, 12.8 MB), so a rerun of this
exact sweep, including the added thresholds, replays from the cache with no paid
calls. Checked: a second run made 0 paid requests (3,832 hits) and produced an
identical summary. Adding new thresholds costs only their rerank asks; any intake change misses
the cache and pays full price.
