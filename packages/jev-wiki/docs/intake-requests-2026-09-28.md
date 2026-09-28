# Requests that carry personal facts — 28 September 2026

After [preference-misses-2026-09-28.md](preference-misses-2026-09-28.md), two
single-session-preference questions still missed on the 30-question live sweep
(`examples/longmemeval.py --per-type 5 --seed 0`, model `jev-1.13.0`). In both, the
answer sits inside a request ("organize my tech accessories, like my new portable
power bank", "inspiration for a new bedroom dresser to replace my old one"). JEV
kept the sentence but filed its kind as `uncertain` (0.73–0.82), because the
`uncertain` description listed "question".

## Change (rubric `wiki-v3`)

- `uncertain` now reads "Hypothesis, speculation, or ambiguous statement; a question
  or request only when it states nothing definite about the speaker or their world".
- The kind question adds: "Judge what it states, not its sentence form: a question or
  request that mentions the speaker's possessions, plans, habits, or situation asserts
  that fact."

`RUBRIC_VERSION` is `wiki-v3`; relations stay on `wiki-v1`. The keep threshold (0.82),
the kind gate (0.70) and the rerank cut are unchanged. Stores that already ran intake
keep their old statuses, because intake does not reprocess.

The harness gains `--types`, which keeps only the named question types from the seeded
selection, so a subset run asks exactly the questions of the full sweep.

## Subset first (preference + single-session-user, 10 questions)

Both labelled answer turns became active (0 → 1 for 09d032c9 and 57f827a0).
57f827a0 now hits lexically; 09d032c9's claim is active but "battery life on my phone"
does not rank "portable power bank", so that miss is now in ranking, not intake. No
question lost recall. Cost: 1,184 paid requests, 4.75M input and 1.16M output tokens.

## Full sweep (30 questions)

| At the shipped gate | `wiki-v2` (PR #15) | `wiki-v3` |
| --- | --- | --- |
| Lexical recall@1 / @5 / @10 | 0.57 / 0.70 / 0.73 | **0.63 / 0.77 / 0.77** |
| JEV rerank recall@10 | 0.70 | **0.73** |
| Preference, lexical / JEV rerank recall@10 | 0.6 / 0.4 | **0.8 / 0.6** |
| Labelled answer turns overlapped by an active claim | 87% | **91%** |
| Evidence sessions retained | 0.89 | 0.92 |
| Active claims per question | 139 | 155 (+12%) |

No question lost recall in any mode. Gains: 57f827a0 (lexical) and b0479f84 (JEV
rerank; it already reached the shortlist, so this one is rerank variance on a new
shortlist rather than intake). Every other type is unchanged; active claims grew
10–15% in each. Below the shipped gate, keep ≥ 0.65 now gives 0.80 lexical recall@10
(was 0.73) and the keep top choice 0.83 (was 0.73); all stored claims stay at 0.87.

Cost: the full sweep re-asked all intake. With the subset replayed from the cache it
made 2,493 paid requests (10.1M input, 2.5M output tokens), so this change cost 3,677
paid requests, 14.9M input and 3.6M output tokens in all. Results JSON:
`/mnt/project-files/benchmarks/longmemeval-intake-v3-2026-09-28.json`; the responses
are in the shared cache, so a replay is free.

## Still missing

- 09d032c9: the power bank claim is active but the query shares no words with it. The
  optional embedding candidates ([embedding-candidates-2026-09-27.md](embedding-candidates-2026-09-27.md))
  are the likely fix; the live harness runs without them.
- single-session-assistant stays at 0 by design (assistant turns are proposals).
