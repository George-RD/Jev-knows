# Sentence-level intake on LongMemEval — 27 September 2026

The first LongMemEval_S sweep (`examples/longmemeval.py`, 30 questions: `--per-type 5
--seed 0`, no abstention items) found intake, not the keep threshold, losing memories:
JEV judged whole chatty paragraphs as `discard`, taking any personal fact inside a
request with them. Rubric `wiki-v2` changes two things:

1. **Sentence candidates.** Each paragraph is cut at sentence ends and line breaks
   (fragments under 25 characters join a neighbour; sentences over 600 characters are
   cut at whitespace). Up to eight neighbouring sentences share one state as context.
2. **Keep wording.** The question asks whether a personal assistant would want to
   remember the sentence later, and the `keep` option names personal facts, plans,
   preferences and commitments, "even when mentioned in passing or inside a question
   or request".

The default keep threshold stays at 0.82. Model `jev-1.13.0`, cache disabled. The
three runs used the same 30 questions and the provider from before PR #3's final
score-drift fix; the after run lost 14 of 360 reranks to that validation (excluded
from JEV averages, as the harness does).

## Results (recall_any@10 over sessions, `wiki_lexical` unless noted)

BM25 over raw user turns scores 0.93 on these questions. The 30-question "before" column matches
the full 346-question baseline in [longmemeval-2026-09-27.md](longmemeval-2026-09-27.md)
(shipped 0.01, keep top choice 0.59, every stored claim 0.88). The sweep ran before
the TypeSafe account ran out of credits.

| Acceptance policy | Before (paragraphs, v1 wording) | Sentences only (v1 wording) | After (`wiki-v2`) |
| --- | --- | --- | --- |
| shipped (keep ≥ 0.82) | 0.00 | 0.03 | **0.63** (JEV rerank 0.60) |
| keep ≥ 0.6 | 0.10 | 0.10 | 0.70 |
| keep ≥ 0.4 | 0.27 | 0.27 | 0.70 |
| keep top choice | 0.47 | 0.37 | 0.73 |
| every stored claim | 0.80 | 0.60 | 0.87 |

| Retention at the shipped gate | Before | Sentences only | After |
| --- | --- | --- | --- |
| Candidates activated | 4 of about 9,700 (0.04%) | 0.3% | 2,754 of 20,502 (13.4%) |
| Evidence sessions with an active claim | 2.1% | 6.4% | 83% |
| Labelled answer turns overlapped by an active claim | 0% | 0% | 71% |

The sentences-only ablation deferred 367 of 1,439 sources on provider validation
failures, so its retention is a lower bound, but among the sources it did ingest the
shipped gate kept 0.3% of candidates. The wording change carries the gain; splitting
lets the gain stay precise (a kept claim is one sentence, not a paragraph).

At the shipped gate, recall is 1.0 for knowledge-update, multi-session and
single-session-user questions, 0.8 for temporal reasoning, and 0.0 for
single-session-assistant (the answer is in assistant turns, which are never active
evidence) and single-session-preference (the preference is implied, and the question
shares few words with it).

## Cost

| Per 30-question run | Before | After |
| --- | --- | --- |
| JEV questions | 29,060 | 69,846 |
| Input / output tokens | 5.88M / 1.51M | 13.63M / 3.64M |
| Requests | 3,019 | 3,592 |

Intake costs about 2.3 times as much, because there are about twice as many
candidates and each still gets keep, kind and topic questions. Tokens per question
are unchanged, at about 200 input and 50 output.

## Open

- Recall at the shipped gate (0.63) is now close to the keep-top-choice ceiling
  (0.73), so the threshold decision matters less than before; it remains George's.
- The JEV rerank still loses a little recall against lexical order at every policy.
- Preference questions need candidate expansion (continuation item 1), not intake.
