# Live JEV evaluation — 27 September 2026

First run of `examples/evaluate.py` against the real service. Environment: Linux,
Python 3.11.15, `https://api.typesafe.ai/v1/systemone`, requested model `jev-1.13.0`
(the only version served: `jev-latest` and `jev-preview` both resolved to `jev-1.13.0`).
Corpus `jev-wiki-synthetic-v1`, `k=3`, cache disabled, three clean trials per variant.

## Service behaviour

- Every request succeeded on the first attempt (no retries). Median decision call
  about 0.28 s including network; ingesting all ten sources took 3.1–3.7 s.
- One ingestion run costs about 7.7k input and 1.7k output tokens (30 questions).
  A reranked query costs about 500 input and 35 output tokens.
- Choice probabilities come back rounded to two decimals. With seven topic options
  they summed to 0.99 or 1.01 in about 5% of responses (4 of 80 in a probe), which the
  provider's 0.005 tolerance rejected, deferring the whole source. The first
  unpatched run stopped at source 7 for this reason. The tolerance now scales with
  the option count (`test_two_decimal_rounding_drift_is_accepted`).
- `confidence` on a choice answer is not the chosen option's probability. Observed
  pairs (probability → confidence): 0.86 → 0.79, 0.73 → 0.59, 0.54 → 0.31,
  0.49 → 0.24. It behaves like a margin over the runner-up.

## Results with the shipped thresholds

The engine only activates a claim when `keep` confidence is at least 0.82. No source
reached that in trial 1; only `mira-updates` (0.79–0.86) did in trials 2 and 3.
Everything else stayed in the review queue, so retrieval had almost nothing to find.

| Trial | Accepted sources | Answerable recall@3 (lexical / JEV) | Empty-query accuracy (lexical / JEV) |
| --- | --- | --- | --- |
| 1 | 0 of 10 | 0.00 / 0.00 | 1.0 / 1.0 |
| 2 | 1 of 10 | 0.14 / 0.14 | 1.0 / 1.0 |
| 3 | 1 of 10 | 0.14 / 0.14 | 1.0 / 1.0 |

These numbers measure the acceptance gate, not retrieval.

## Sensitivity run: trust JEV's top choice

To see retrieval behind the gate, the same harness was run with `_confidence`
patched to 1.0 (a claim is active whenever JEV's top `keep` choice is `keep`). This is
an experiment, not a proposed setting.

| Trial | Accepted sources | Answerable recall@3 (lexical / JEV) | Empty-query accuracy (lexical / JEV) |
| --- | --- | --- | --- |
| 1 | 5 of 10 | 0.43 / 0.43 | 0.5 / 1.0 |
| 2 | 5 of 10 | 0.43 / 0.43 | 0.5 / 1.0 |
| 3 | 4 of 10 | 0.43 / 0.43 | 0.5 / 1.0 |

- JEV reranking never lost a relevant source and removed every distractor: the
  "no answer with anchor" query returned two wrong Kestrel claims lexically and none
  after reranking; the database query dropped the unrelated permission claim.
- Even as a top choice, JEV put `printer-power`, `menu` and both Cedar sign-off
  sources in review, and chose `discard` for `backup-procedure`. So the conflict
  pair was never accepted and conflict linking could not be tested.
- `zero-overlap-synonym` stays a miss whenever `petrol-vehicles` is accepted: no
  lexical candidate is offered, as the evaluation doc predicts.

## What this means

1. The keep gate is miscalibrated for live confidences. A threshold on the chosen
   option's probability, or a lower confidence cut-off chosen from a labelled set,
   is needed before any retrieval number is meaningful.
2. The `keep` question itself rejects useful facts and procedures. The rubric
   wording (or asking per-kind retention questions) should be revisited before
   tuning thresholds against it.
3. Reranking already does its job on this corpus: precision up, recall unchanged.
4. Ten sources and nine queries cannot calibrate anything. A public memory
   benchmark is needed for the next measurement.
