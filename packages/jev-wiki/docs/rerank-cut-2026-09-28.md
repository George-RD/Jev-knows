# JEV rerank cut-off sweep — 28 September 2026

The live sweep in [longmemeval-live-2026-09-28.md](longmemeval-live-2026-09-28.md) found
that the JEV rerank returns about 1.1 claims per question at the shipped keep gate,
because `Engine.recall` drops every shortlisted claim JEV scores below 1.5 (on its 0–3
scale, 0 Unrelated to 3 Direct evidence, returned as an expected value). This sweep
measures other cuts on the same 30 questions.

`Engine.recall` now takes `min_relevance` (default 1.5, unchanged) and `backfill`
(default off), and the CLI takes `--min-relevance` and `--backfill`:

- **cut t**: keep claims scoring at least t, best score first (shipped: t = 1.5).
- **reorder**: t = 0, so every shortlisted claim is kept, in JEV score order.
- **backfill t**: claims scoring at least t first, in score order, then the rest of the
  shortlist in lexical order. Nothing lexical recall found is lost.

Every variant sends the identical rerank request, so the whole sweep replayed from the
shared JEV cache: 6,231 cache hits and 1 paid request (1,967 input tokens; its cache write failed once,
so the cache still holds 3,716 entries). The
shipped-cut numbers match PR #12's exactly. Harness: `examples/longmemeval.py
--per-type 5 --seed 0` (the `wiki_jev_*` modes). Full results:
`/mnt/project-files/benchmarks/longmemeval-rerank-cut-2026-09-28.json`.

## At the shipped keep gate (0.82)

| Rerank | Claims returned | recall_any@1 | @5 | @10 |
| --- | --- | --- | --- | --- |
| none (lexical order) | 18.5 | 0.57 | 0.67 | 0.67 |
| cut 1.5 (shipped) | 1.1 | 0.63 | 0.63 | 0.63 |
| cut 1.0 | 1.5 | **0.67** | 0.67 | 0.67 |
| cut 0.5 | 2.5 | 0.67 | 0.67 | 0.67 |
| cut 0.1 | 5.7 | 0.67 | 0.67 | 0.67 |
| reorder (cut 0) | 18.5 | 0.67 | 0.67 | 0.67 |
| backfill 1.5 | 18.5 | 0.63 | 0.67 | 0.67 |
| backfill 0.5 | 18.5 | 0.67 | 0.67 | 0.67 |

Any cut at or below 1.0 fixes the loss: recall@10 is back to lexical's 0.67, and
recall@1 is 0.67 against lexical's 0.57. All three changed questions are multi-session:
JEV puts an evidence session first in two where lexical order does not, and the shipped
cut drops the only evidence claim in the third (`gpt4_2ba83207`, scored between 1.0 and
1.5).

## At looser keep gates

When intake keeps more, the ranking has more to work with and the low cuts pull ahead
of lexical order.

| Keep gate | Lexical @1 / @10 | cut 1.5 (shipped) | cut 0.5 | cut 0.1 | reorder | backfill 1.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.82 (shipped) | 0.57 / 0.67 | 0.63 / 0.63 | 0.67 / 0.67 | 0.67 / 0.67 | 0.67 / 0.67 | 0.63 / 0.67 |
| 0.65 | 0.57 / 0.73 | 0.67 / 0.67 | 0.70 / 0.73 | **0.73 / 0.77** | 0.73 / 0.77 | 0.67 / 0.73 |
| keep top choice | 0.60 / 0.73 | 0.73 / 0.73 | 0.73 / 0.77 | 0.73 / 0.77 | 0.73 / 0.77 | 0.73 / 0.73 |
| every stored claim | 0.77 / 0.87 | 0.80 / 0.80 | 0.83 / 0.87 | 0.83 / 0.87 | 0.83 / 0.87 | 0.83 / 0.87 |

## Reading it

- **cut 0.1 wins.** It equals full reorder on every metric at every keep gate, while
  returning about 6 claims instead of 18–20, because JEV scores almost every irrelevant
  claim at 0.00–0.03. Across all ten keep gates it is never worse than lexical order,
  beats it at recall@1 everywhere (+0.07 to +0.17) and at recall@10 at eight of them
  (+0.03, one question).
- **Backfill is the weaker way to keep recall.** Ordering the low-scoring rest
  lexically, instead of by JEV score, gives back part of the recall@1 gain. Backfill
  guarantees recall never falls below lexical order, which a cut can only lose on an
  evidence claim JEV scores below the cut (none here at 0.1).
- The shipped 1.5 cut is the only setting that loses to lexical order at recall@10.
- The gains are one to three questions out of 30 (one question is 0.033), so this is
  direction, not precision. The rerank itself cannot fix single-session-preference
  (0.2) or single-session-assistant (0.0); those misses are upstream.

## Default

The default cut stays 1.5 until George decides; `min_relevance=0.1` (CLI
`--min-relevance 0.1`) is opt-in. The keep threshold (0.82) is unchanged.
