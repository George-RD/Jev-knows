# Where preference answers were lost — 28 September 2026

On the 30-question live LongMemEval_S sweep (`examples/longmemeval.py --per-type 5
--seed 0`, model `jev-1.13.0`, [longmemeval-live-2026-09-28.md](longmemeval-live-2026-09-28.md)),
single-session-preference recall@10 was 0.2: four of five questions missed at the
shipped gate. Offline, with every candidate kept, the same type scores 0.967, so the
loss is in intake, before ranking.

## Diagnosis: the kind gate, not the keep threshold

Replaying the five questions from the response cache and listing every candidate that
overlaps a labelled answer turn: JEV answered `keep` for every answer sentence, with
keep confidence 0.85–0.98, above the 0.82 threshold. Each was held in review by the
**kind** gate (`kind != uncertain` and kind confidence ≥ 0.70), in two ways.

| Question | Answer sentence (abridged) | Kind probabilities | Cause |
| --- | --- | --- | --- |
| 0a34ad58 Tokyo | "I have downloaded the TripIt app to stay organized…" | fact 0.72, decision 0.18, uncertain 0.02 | margin 0.67 < 0.70 |
| 0a34ad58 Tokyo | "…visit the Tsukiji Fish Market." | decision 0.49, commitment 0.34, uncertain 0.04 | margin 0.38 |
| b0479f84 documentaries | "I've been watching a lot of documentaries lately, especially on Netflix." | preference 0.60, fact 0.39, uncertain 0.01 | margin 0.52 |
| 57f827a0 bedroom | "I'm looking for … inspiration for a new bedroom dresser to replace my new one…?" | uncertain 0.73, preference 0.26 | request labelled `uncertain` |
| 09d032c9 battery | "…organize my tech accessories, like my new portable power bank…" | uncertain 0.82, preference 0.16 | request labelled `uncertain` |

1. **Margin misread (fixed here).** JEV's choice `confidence` is a margin between the
   top two options, not the chosen option's probability. A personal fact JEV cannot
   file as fact versus preference gets a low margin although it is plainly not
   speculation, which is the only thing the kind gate is there to stop.
2. **Requests labelled `uncertain` (not fixed).** The `uncertain` kind reads
   "Hypothesis, speculation, question, or ambiguous statement", so a request that
   carries a personal fact is filed as a question. Fixing that means rewording the
   kind rubric, which misses the cache and re-pays all intake.

## Change

`definite_kind()` in `engine.py` gates on `1 - P(uncertain) ≥ 0.70` when the kind
answer carries probabilities (JEV answers always do) and falls back to the top
choice's confidence otherwise. The top choice still must not be `uncertain`. The keep
threshold (0.82), the 0.70 number and the rerank cut are unchanged. The harness's
policies use the same function.

Follow-up (PR #16, after review of #15): the distribution is trusted only when it is
complete and consistent (every kind present, finite, summing to one within rounding,
the chosen kind on top), otherwise the gate falls back to the top choice's confidence.
Completed sources are never re-asked, so `Engine.reclassify()`, run by every `worker`
pass, replays stored decisions for claims intake left in review and promotes those the
current gate admits. It leaves alone claims of superseded or forgotten sources, claims
from an older rubric, and any claim a caller has updated. Checked on 2,459 stored claims
replayed from the cache: the stricter validation gates every real JEV answer the same
way, so the results below stand.

A sample of claims the change activates (25 of 147 in three questions) is all
first-person facts, plans and preferences, for example "I've been playing the guitar
for about 6 months now", "I've got a baby grand that needs to be moved", "I am
definitely adding Yakushima and Tottori to my list now". The borderline ones are
monologue-style transcript fragments that carried `uncertain` 0.20–0.29.

## Results (same 30 questions, replayed from the response cache)

| At the shipped gate | Before | After |
| --- | --- | --- |
| Lexical recall@10 | 0.67 | **0.73** |
| JEV rerank recall@10 | 0.63 | **0.70** |
| Preference, lexical / JEV rerank recall@10 | 0.2 / 0.2 | **0.6 / 0.4** |
| Labelled answer turns overlapped by an active claim | 69% | **87%** |
| Evidence sessions retained | 0.83 | 0.89 |
| Active claims per question | 91 | 139 (+52%) |

No question lost recall in any mode. Gains: 0a34ad58 and b0479f84 (lexical),
0a34ad58 and multi-session gpt4_2ba83207 (JEV rerank). b0479f84 reaches the JEV
shortlist but falls to the `relevance >= 1.5` cut (see PR #13). Recall@1 is
unchanged at 0.57. The shipped gate now matches or beats what keep ≥ 0.65 gave
before, and the keep sweep below 0.82 adds less than it did (0.65 gives no further
lexical gain).

Cost: intake replayed from the cache; the only paid requests were 383 reranks over
the changed shortlists (1.22M input tokens). Results JSON:
`/mnt/project-files/benchmarks/longmemeval-kind-gate-2026-09-28.json`.

## Still missing

57f827a0 and 09d032c9 need the `uncertain` kind reworded so a question or request is
filed by what it reveals about the speaker. That is a rubric change (a new
`RUBRIC_VERSION`): it re-asks all intake, about 3.7k requests and 14M input tokens for
this sweep.
