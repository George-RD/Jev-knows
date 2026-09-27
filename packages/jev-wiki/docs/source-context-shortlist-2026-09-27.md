# Source-context shortlist, offline LongMemEval check (2026-09-27)

Preference questions were the weakest type after the BM25 shortlist
([bm25-shortlist-2026-09-27.md](bm25-shortlist-2026-09-27.md)): 0.70 recall_any@10
with every candidate kept. This change lifts each matching claim by how well its whole
source matches the query. It was measured without JEV, because the TypeSafe account
was out of credits.

## What was wrong

The misses were not only a vocabulary gap. A request such as "I've been feeling a bit
stuck with my paintings lately. Do you have any ideas on how I can find new
inspiration?" shares its request words ("any", "ideas", "do", "you", "have") with
requests in every other session, so the top claims were "Do you have any ideas on how
to categorize and display my books?" and similar. The session that answers it keeps
coming back to paintings, but no single sentence outscores the filler matches.

## Change

`Engine.recall` scores each source as one BM25 document (its active claims plus its
title) and multiplies each matching claim's own BM25 score by
`1 + 0.5 * source score / best source score`. The shortlist is ordered by that
`context_score`; `lexical_score` stays the plain BM25 score.

- The lift is proportional to the claim's own match, so a claim can only overtake
  claims that score less than 1.5 times as well on their own. Unmatched claims stay
  out, and a claim that matches only through its source title gets no lift.
- The twelve best plain-BM25 claims are always taken into the 24-claim shortlist first,
  byte budget included. With only a few sources, source-level BM25 cannot tell filler
  words from topic words, so a source that repeats "any ideas for my" could otherwise
  push the one claim holding a rare query term out before JEV reranks. In offline or
  degraded mode such a claim is still returned in context order, which can put it
  below the lifted claims.

## Choosing the lift

Variants were compared with a fast replica of the offline scorer (it reproduces the
`main` numbers exactly) over all 470 questions:

| Variant | @1 | @10 | Preference @10 | Notes |
|---|---|---|---|---|
| BM25 (`main`) | 0.804 | 0.953 | 0.700 | |
| Drop request filler words from the query | | | 0.733 | multi-session fell to 0.959 |
| Additive blend, 0.5 of normalised source score | 0.843 | 0.962 | 0.800 | lost one multi-session question; a weak match could outrank a strong one |
| Source score with claim-level IDF, ×(1 + 0.5 s) | 0.815 | 0.957 | 0.767 | |
| Multiplicative ×(1 + 0.5 s), shipped | 0.832 | 0.962 | 0.800 | no question lost |
| Multiplicative ×(1 + 1.0 s) | 0.843 | 0.964 | 0.800 | |
| Multiplicative ×(1 + 2.0 s) | 0.845 | 0.962 | 0.800 | |

A weight of 1.0 scores slightly higher at @1, but it lets a filler-heavy source
overtake a claim that matches up to twice as well, which failed a code-review case.
0.5 keeps that margin at 1.5 and loses nothing at @10.

Adding stored `preference` claims to recommendation queries, as first proposed,
cannot be measured offline: the scorer's stub labels every claim `fact`. The source
lift needs no kind label and helps other types too.

## Results

[`examples/longmemeval_offline.py`](../examples/longmemeval_offline.py) on the final
code, all 470 non-abstention questions of LongMemEval_S cleaned:

| Shortlist | recall_any@1 | @3 | @5 | @10 | recall_all@10 |
|---|---|---|---|---|---|
| BM25 (`main`) | 0.804 | 0.894 | 0.928 | 0.953 | 0.849 |
| BM25 + source context | 0.832 | 0.909 | 0.940 | 0.962 | 0.864 |

recall_any@10 by question type:

| Type | BM25 | + source context |
|---|---|---|
| knowledge-update | 1.000 | 1.000 |
| multi-session | 0.975 | 0.975 |
| single-session-assistant | 0.946 | 0.946 |
| single-session-preference | 0.700 | 0.800 |
| single-session-user | 1.000 | 1.000 |
| temporal-reasoning | 0.945 | 0.953 |

No type fell, and no single question was lost at @10. Four were gained: three
preference questions and one temporal-reasoning question.

30-question sweep (`--per-type 5 --seed 0`): recall_any@1 0.733 to 0.767, @5 0.867 at
both, @10 0.867 to 0.900. Preference questions went from 0.4 to 0.6 at @10; every other
type was unchanged.

## Still open

Six preference questions still miss, such as "What should I serve for dinner this
weekend with my homegrown ingredients?", where the evidence says "basil", "mint" and
"cherry tomatoes" but never "dinner" or "homegrown". Closing that gap needs candidate
expansion that is not lexical (JEV topic routing or embeddings).

These numbers measure candidate generation with every candidate kept. A live rerun of
the 30-question sweep with JEV intake and reranking is pending on TypeSafe credits.
