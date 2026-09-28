# Rerank cut with embedding candidates — 28 September 2026

The live sweep with embedding candidates on ([embedding-live-2026-09-28.md](embedding-live-2026-09-28.md))
scored the JEV rerank at the shipped `relevance >= 1.5` cut only. The rerank cut sweep
([rerank-cut-2026-09-28.md](rerank-cut-2026-09-28.md)) was run before embeddings. This run
combines them: the 30-question live sweep (`examples/longmemeval.py --per-type 5 --seed 0`,
model `jev-1.13.0`, rubric `wiki-v3`, embedder `minishlab/potion-retrieval-32M`) with every
rerank variant, so the cut can be decided with embeddings on. No defaults change: the keep
gate stays 0.82 and the rerank cut stays 1.5.

Run:

    JEV_WIKI_EMBEDDING_MODEL=default python examples/longmemeval.py \
        --data longmemeval_s_cleaned.json --per-type 5 --seed 0 --workers 10 \
        --cache-dir /mnt/project-files/jev-cache/longmemeval --output report.json

## Results at the shipped keep gate (0.82), session recall_any

| Recall | @1 | @5 | @10 | Claims returned |
| --- | --- | --- | --- | --- |
| Offline (hybrid candidates, no rerank) | 0.70 | 0.83 | 0.83 | up to 20 |
| JEV rerank, cut 1.5 (shipped) | 0.70 | 0.70 | 0.70 | 1.7 |
| JEV rerank, cut 1.0 | 0.80 | 0.80 | 0.80 | 2.4 |
| JEV rerank, cut 0.5 | 0.80 | 0.83 | 0.83 | 4.6 |
| **JEV rerank, cut 0.1** | **0.80** | **0.83** | **0.83** | 9.2 |
| JEV rerank, full reorder (cut 0) | 0.80 | 0.83 | 0.83 | 20 |
| Cut 1.5 with backfill | 0.73 | 0.83 | 0.83 | 20 |

Cut 0.1 matches a full reorder again, and it is the only setting that is at least as good
as offline order on every metric: it keeps the hybrid candidates' recall@10 (0.83) and
lifts recall@1 from 0.70 to 0.80. The shipped cut loses 4 of the 25 questions the
candidates found.

| Type (shipped gate) | Offline @1 / @10 | Cut 1.5 @10 | Cut 0.1 @1 / @10 |
| --- | --- | --- | --- |
| knowledge-update | 1.0 / 1.0 | 1.0 | 1.0 / 1.0 |
| multi-session | 0.8 / 1.0 | 1.0 | 1.0 / 1.0 |
| single-session-assistant | 0 / 0 | 0 | 0 / 0 |
| single-session-preference | 0.6 / 1.0 | 0.4 | 0.8 / 1.0 |
| single-session-user | 1.0 / 1.0 | 1.0 | 1.0 / 1.0 |
| temporal-reasoning | 0.8 / 1.0 | 0.8 | 1.0 / 1.0 |

Questions where the cut changes the result (cut 1.5 → 0.1, recall@10): 09d032c9 (power
bank), 0a34ad58 (Tokyo tips), 57f827a0 (bedroom furniture) and gpt4_e061b84f (sports
event order) all go 0 → 1. No question is lost at 0.1.

Lower keep gates, cut 1.5 → cut 0.1 (recall@1 / @10):

| Keep gate | Offline | Cut 1.5 | Cut 0.1 |
| --- | --- | --- | --- |
| 0.65 | 0.70 / 0.83 | 0.70 / 0.70 | 0.80 / 0.83 |
| JEV's top keep choice | 0.70 / 0.87 | 0.73 / 0.73 | 0.83 / 0.87 |
| All stored claims | 0.77 / 0.97 | 0.83 / 0.83 | 0.90 / 0.97 |

With embeddings on, lowering the keep gate to 0.65 no longer gains anything at either cut;
the candidates already find what 0.65 used to add.

## Cost

One paid TypeSafe request (5.8k input tokens). Every other answer (3,743 requests, 15.1M
input tokens) replayed from the shared cache, because all rerank variants share one
request per question and the shortlists match the embeddings sweep. The single miss
followed one unreadable cache entry.

Results JSON: `/mnt/project-files/benchmarks/longmemeval-embeddings-rerank-cut-2026-09-28.json`.
