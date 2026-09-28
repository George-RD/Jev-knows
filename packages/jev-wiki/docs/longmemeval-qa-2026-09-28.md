# LongMemEval end-to-end QA (2026-09-28)

The retrieval harnesses score whether a labelled session reaches the top k. That is
not what an agent needs. An agent needs the recalled claims to be enough to answer.
`examples/longmemeval_qa.py` measures that: a reader model answers each LongMemEval_S
question from what jev-wiki recalls, and a judge grades the answer with LongMemEval's
own per-type judge prompts.

## Setup

- All 500 LongMemEval_S questions, abstention included.
- Intake: the offline keep-everything stub (every sentence claim active). No TypeSafe
  calls. This is recall plus answering under an ideal intake, not the live keep policy.
- Recall: `Engine.recall(offline=True, limit=20, max_chars=20_000)` with
  `JEV_WIKI_EMBEDDING_MODEL=default`. Each claim is shown with the date of its chat.
- Reader `gpt-oss:120b`, judge `glm-5.3`, both on Ollama Cloud at temperature 0.
  LongMemEval's published numbers use GPT-4o, so compare these only with runs of
  this harness that use the same models.
- `oracle` gives the reader the user turns of the labelled evidence sessions instead,
  which is the ceiling for a store that keeps only user turns. `none` gives it nothing.

## Results

| Type | n | none (30-question sample) | wiki | oracle |
|---|---:|---:|---:|---:|
| single-session-user | 64 | 0.00 | 0.953 | 1.000 |
| single-session-assistant | 56 | 0.20 | 0.357 | 0.411 |
| single-session-preference | 30 | 0.20 | 0.600 | 0.727 (22 graded) |
| multi-session | 121 | 0.00 | 0.711 | 0.884 |
| knowledge-update | 72 | 0.00 | 0.931 | 0.944 |
| temporal-reasoning | 127 | 0.00 | 0.819 | 0.945 |
| abstention | 30 | 1.00 | 0.700 | 0.621 |
| **all** | 500 | 0.167 | **0.754** | **0.847** |
| answerable only | 470 | 0.074 | 0.757 | 0.861 |

Nine oracle rows stayed ungraded after retries (Ollama Cloud's concurrency limit).
Eight of them are preference questions. They are left out of that column's accuracy.

## What it shows

- **Session recall no longer predicts answers.** A labelled session is in the top 10
  for 492 of 500 questions, but the answer is right for only 0.764 of those. Session
  recall@10 of 0.98 overstates how good the memory is for an agent.
- **The losses are aggregation and time.** Paired against the oracle, 25 multi-session
  and 19 temporal-reasoning questions go from right to wrong (61 overall; 16 go the
  other way). These questions need every mention, like each art event or the user's own
  age next to their parents'. The 20-claim limit returns some of the mentions, and the
  reader then counts or computes from a partial set. Both contexts have a median of
  about 2.9k characters, so the limit is the claim count, not the size budget.
- **Knowledge updates and single facts are fine.** knowledge-update is within one point
  of the oracle, and single-session-user within five.
- **Assistant-turn questions fail by design.** Only user turns are ingested, so both
  columns score about 0.4, which is what the reader guesses from the user's side.
- **Abstention is slightly better with wiki recall.** Sparse evidence makes the reader
  more willing to say it doesn't know.

## Next

- Try a larger or adaptive recall limit for counting and "how many" questions, and
  rerun this harness. Done: [aggregate recall](aggregate-recall-2026-09-28.md) lifts
  the overall score to 0.778 and multi-session to 0.818.
- Run it with live intake (the shipped keep policy) to measure what intake loses on top
  of recall. That costs the same JEV calls as the retrieval sweep.
- For conflict resolution beyond LongMemEval's knowledge-update type, MemoryAgentBench's
  conflict-resolution split is the next benchmark worth adding.

## Reproduce

```bash
pip install -e "packages/jev-wiki[embed]"
export JEV_WIKI_EMBEDDING_MODEL=default OLLAMA_API_KEY=...
python packages/jev-wiki/examples/longmemeval_qa.py --data longmemeval_s_cleaned.json \
    --per-type 0 --workers 4 --output wiki.json
# rows that hit rate limits: rerun only those
python packages/jev-wiki/examples/longmemeval_qa.py --data longmemeval_s_cleaned.json \
    --per-type 0 --workers 4 --resume wiki.json --output wiki.json
```

The numbers above answered each question once. The harness now answers each question
three times from the same notes and grades it by majority vote (`--samples`, default 3),
because single answers flip about 6% of verdicts between identical runs; `--samples 1`
reproduces the runs above at a third of the reader and judge calls.

The wiki run took about 25 minutes at 12 workers. Ollama Cloud limits concurrent
requests, so more workers produce 429s rather than speed.
