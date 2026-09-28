# Embedding candidates on the live sweep — 28 September 2026

The optional embedding candidates ([embedding-candidates-2026-09-27.md](embedding-candidates-2026-09-27.md),
PRs #8/#9) had only been scored offline. This reruns the 30-question live sweep
(`examples/longmemeval.py --per-type 5 --seed 0`, model `jev-1.13.0`, rubric `wiki-v3`)
with them on and compares it with the `wiki-v3` baseline
([intake-requests-2026-09-28.md](intake-requests-2026-09-28.md)). No defaults change.

## Harness change

`--embedding-model NAME` (default `$JEV_WIKI_EMBEDDING_MODEL`; `default` is
`minishlab/potion-retrieval-32M`) passes a `StaticEmbedder` to each question's `Engine`,
so both `wiki_lexical` (offline recall, now `hybrid` candidates) and `wiki_jev` use the
embedding candidates. The model loads once before any paid request and a load failure
stops the run, so a sweep meant to measure embeddings can't quietly score lexical recall.
The report records `embedding_model`. `bm25_claims*` are unchanged.

Run:

    JEV_WIKI_EMBEDDING_MODEL=default python examples/longmemeval.py \
        --data longmemeval_s_cleaned.json --per-type 5 --seed 0 --workers 10 --output report.json

## Results (session recall_any@10)

| At the shipped gate (0.82) | Baseline (lexical) | Embeddings on |
| --- | --- | --- |
| Offline recall@1 / @5 / @10 | 0.63 / 0.77 / 0.77 | **0.70 / 0.83 / 0.83** |
| JEV rerank recall@10 | 0.73 | 0.70 |
| JEV rerank claims returned per question | 1.67 | 1.73 |

| Type (shipped gate) | Offline @10 | JEV rerank @10 |
| --- | --- | --- |
| knowledge-update | 1.0 → 1.0 | 1.0 → 1.0 |
| multi-session | 1.0 → 1.0 (@1 0.6 → 0.8) | 1.0 → 1.0 |
| single-session-assistant | 0 → 0 | 0 → 0 |
| single-session-preference | 0.8 → **1.0** | 0.6 → 0.4 |
| single-session-user | 1.0 → 1.0 | 1.0 → 1.0 |
| temporal-reasoning | 0.8 → **1.0** | 0.8 → 0.8 |

Below the gate, offline recall@10 is 0.83 at keep ≥ 0.65 (was 0.80), 0.87 at the keep
top choice (was 0.83) and 0.97 over all stored claims (was 0.87).

Question-level changes at the shipped gate:

- 09d032c9 (preference, "battery life on my phone"): the active power bank claim now
  reaches the offline top 10. This was the miss left by `wiki-v3`.
- gpt4_e061b84f (temporal): now found offline.
- 0a34ad58 (preference, Tokyo tips): **lost under the JEV rerank** (1 → 0). The answer
  claim is still in the shortlist (offline @10 hits), but with the new shortlist JEV
  scored every candidate below the `relevance >= 1.5` cut and returned nothing. The
  baseline also only kept one claim here.

No question lost offline recall. The JEV rerank's cut, not the candidates, now decides
the live result: hybrid candidates reach 0.83 but the rerank returns about 1.7 claims per
question and scores 0.70. That matches [PR #13's cut sweep](https://github.com/George-RD/Jev-knows/pull/13),
where a 0.1 cut beat the shipped 1.5.

## Cost

Intake replayed entirely from the shared cache. Only reranks over new shortlists were
paid: **254 paid requests, 0.81M input and 0.10M output tokens** (3,578 cache hits
saved 14.6M input tokens). Wall time 11.7 minutes with 10 workers. Results JSON:
`/mnt/project-files/benchmarks/longmemeval-embeddings-2026-09-28.json`; its responses
are in the shared cache, so a replay is free.

## What the data suggests (decisions are George's)

- Embedding candidates are a clear win for candidate generation (+2 questions offline,
  no losses, preference to 1.0) at low cost: a 129 MB model and no paid calls.
- Turning them on by default only pays off under the JEV rerank if the rerank cut also
  moves; with the 1.5 cut the live score is 0.70 vs 0.73, a one-question swing on a
  small sample.
