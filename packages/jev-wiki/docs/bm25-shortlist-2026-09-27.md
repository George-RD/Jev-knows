# BM25 recall shortlist, offline LongMemEval check (2026-09-27)

`Engine.recall` used to rank candidates by the number of query words a claim shares.
It now ranks them by Okapi BM25 (k1 1.2, b 0.75) over active claims plus source titles,
with the same bounds (24 claims, 14 KB). This note measures the change without JEV:
the TypeSafe account was out of credits, so no live call was made.

## Method

[`examples/longmemeval_offline.py`](../examples/longmemeval_offline.py) ingests each
question's haystack through the real engine with a stub provider that keeps every
sentence candidate, activates them all, and scores `Engine.recall(offline=True)` by
session recall. It was run from `main` (overlap count) and from this change (BM25),
on LongMemEval_S cleaned, all 470 non-abstention questions and the usual 30-question
sweep (`--per-type 5 --seed 0`).

Keeping every candidate is an upper bound on retention, close to the live harness's
`all_stored` policy but not identical (that policy uses JEV's stored claims and 346
valid questions). The numbers measure candidate generation only, not intake or JEV
reranking.

## Results

All 470 questions:

| Shortlist | recall_any@1 | @5 | @10 | recall_all@10 |
|---|---|---|---|---|
| Overlap count (`main`) | 0.738 | 0.874 | 0.917 | 0.777 |
| BM25 | 0.804 | 0.928 | 0.953 | 0.849 |

recall_any@10 by question type:

| Type | Overlap count | BM25 |
|---|---|---|
| knowledge-update | 1.000 | 1.000 |
| multi-session | 0.975 | 0.975 |
| single-session-assistant | 0.768 | 0.946 |
| single-session-preference | 0.567 | 0.700 |
| single-session-user | 0.984 | 1.000 |
| temporal-reasoning | 0.929 | 0.945 |

30-question sweep: recall_any@1 0.567 to 0.733, @5 0.800 to 0.867, @10 0.867 at both.
On those 30, preference questions fall from 0.6 to 0.4 at @10 while assistant
questions rise from 0.8 to 1.0; over all 470 both improve.

## Reading

BM25 lifts every rank cutoff and every question type that was not already saturated,
matching the live benchmark's gap (all-claims recall@10 0.88 overlap count against
0.93 BM25). Single-session-assistant gains most because their questions repeat rare
words the user said in the session. Preference questions remain the weakest type even
with every candidate retained: the question and the stated preference seldom share
words, which a lexical shortlist cannot bridge.

A live rerun of the 30-question sweep with JEV intake and reranking is pending on
credits.
