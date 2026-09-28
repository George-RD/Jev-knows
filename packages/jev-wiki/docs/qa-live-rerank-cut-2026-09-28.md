# End-to-end answers with the live pipeline: rerank cut and embeddings — 28 September 2026

The rerank cut sweep with embeddings ([embedding-rerank-cut-2026-09-28.md](embedding-rerank-cut-2026-09-28.md))
showed that a JEV rerank cut of 0.1 beats the shipped 1.5 on session recall. This run asks
whether answers improve too. The QA harness (`examples/longmemeval_qa.py`) had only an offline
mode (keep-everything intake, no rerank), so it gains `--live`: JEV intake with the shipped
keep gate (0.82), then recall with the JEV rerank at `--min-relevance`, through the shared
response cache. No defaults change.

Setup: the 30 live-sweep questions (`--per-type 5 --seed 0 --no-abstention`, the same
selection as `longmemeval.py`), `jev-1.13.0`, rubric `wiki-v3`, reader `gpt-oss:120b`, judge
`glm-5.3` (both at temperature 0, so identical notes give identical answers), embedder
`minishlab/potion-retrieval-32M` when on.

    JEV_WIKI_EMBEDDING_MODEL=default python examples/longmemeval_qa.py \
        --data longmemeval_s_cleaned.json --per-type 5 --no-abstention --live \
        --min-relevance 0.1 --workers 4 --output report.json

## Answer accuracy (30 questions, keep gate 0.82)

| Arm | Accuracy | Claims returned | Session recall@10 |
| --- | --- | --- | --- |
| Shipped: lexical, cut 1.5 | 0.600 | 1.6 | 22/30 |
| Embeddings, cut 1.5 | 0.600 | 1.7 | 21/30 |
| Lexical, cut 0.1 | 0.667 | 7.7 | 23/30 |
| **Embeddings, cut 0.1** | **0.667** | 9.2 | 25/30 |

| Type (n = 5) | Shipped | Emb, 1.5 | Lex, 0.1 | Emb, 0.1 |
| --- | --- | --- | --- | --- |
| knowledge-update | 1.0 | 1.0 | 1.0 | 1.0 |
| multi-session | 0.4 | 0.6 | 0.4 | 0.6 |
| single-session-assistant | 0.2 | 0.2 | 0.2 | 0.2 |
| single-session-preference | 0.6 | 0.6 | 0.8 | 0.8 |
| single-session-user | 0.8 | 0.8 | 0.8 | 0.8 |
| temporal-reasoning | 0.6 | 0.6 | 0.8 | 0.6 |

## Question by question

- **Cut 0.1 never loses an answer.** Against cut 1.5 it gains two questions with embeddings
  (57f827a0, 09d032c9) and two without (57f827a0, gpt4_0a05b494), and loses none.
- **Embeddings gain two answers and lose two**, at either cut. Gained: ef9cf60a
  (multi-session) and gpt4_0a05b494 (temporal). Lost: gpt4_7bc6cf22 ("how many days ago did
  I read the March 15th New Yorker"; the notes changed and the reader answered 5 or 17 days
  instead of 12) and 0a34ad58 (Tokyo tips; the claims were recalled but the answer no longer
  used the Suica card and TripIt, so the preference rubric failed it).
- Embeddings plus cut 0.1 against the shipped setting: 4 gained (ef9cf60a, 57f827a0,
  09d032c9, gpt4_0a05b494), 2 lost (0a34ad58, gpt4_7bc6cf22), net +2.

## Reading

Cut 0.1 improves answers as well as recall, with no regressions in this sample. Embeddings
improve session recall (25/30 vs 23/30 at cut 0.1) but not answers: two up, two down. With
30 questions a two-question difference is within noise, so the answer-level case for
turning embeddings on by default is not made here; the case for cut 0.1 is consistent
across recall and answers.

## Cost

Two paid TypeSafe requests across all four arms (8,253 input tokens): the lexical rerank
for 92a0aa75, which no earlier sweep had cached, and one request for 76d63226 in the
lexical cut-0.1 arm, re-sent because its cache entry could not be read (the shared cache
was being written by another session at the time). 13,046 cache hits. Reader and judge
calls went to Ollama Cloud. Extending to more questions needs live intake for questions no
sweep has cached, about 110 requests per question.

Rows: `/mnt/project-files/benchmarks/longmemeval-qa-live-rerank-cut-2026-09-28.json`
(one report per arm).
