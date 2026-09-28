# Live retrieval evaluation

The harness compares lexical retrieval with JEV reranking on the same accepted
corpus. It requires a real `TYPESAFE_API_KEY`. It does not substitute scripted
answers, invent model results, or report lexical fallback as JEV quality.

Run from the repository root after setting `TYPESAFE_API_KEY` in the environment:

```bash
python packages/jev-wiki/examples/evaluate.py --output /tmp/jev-wiki-evaluation.json
```

To retain the generated wiki for inspection, pass a new or empty directory:

```bash
python packages/jev-wiki/examples/evaluate.py \
  --root /tmp/jev-wiki-evaluation-memory \
  --output /tmp/jev-wiki-evaluation.json
```

The default uses a temporary directory and deletes it after the report is created.
An existing nonempty root is refused so prior memories cannot contaminate results.
`--k` defaults to 3. `--model` can override the provider's pinned default. Caching is
disabled for this evaluation; the report records the requested model and observed
versioned response models.

Without a key, the script exits 2 and emits an explicit refusal with no semantic
metrics and zero requests. Incomplete live runs exit 1. A completed run exits 0;
this means the evaluation finished, not that the memory system met a quality gate.

## Protocol

1. Ingest ten synthetic, single-paragraph documents through the real engine and
   JEV provider. The material covers facts, preferences, a decision, a procedure,
   distractors, and two incompatible ownership assertions about the same release.
2. Run bounded maintenance on those accepted claims without supplying labels.
3. Ask nine fixed queries twice: `recall(offline=True)` and live `recall()`. Both
   modes see exactly the same accepted claims. Expected source labels never enter
   ingestion, maintenance, or ranking requests.

Queries are held out from ingestion and were fixed before execution. They are
developer-authored examples, not a statistically independent benchmark. Rejected
or review-only expected sources remain misses: the script reports them explicitly
instead of adjusting labels or manually promoting claims.

## Read the report

| Field | Meaning |
| --- | --- |
| `recall_at_k` | Fraction of expected source keys retrieved; only defined for answerable queries. |
| `expected_empty_correct` | Whether an unanswerable query returns no evidence; undefined for answerable queries. |
| `mean_query_score_at_k_including_empty` | Mean across all valid queries, using recall for answerable queries and 1/0 for correct/incorrect empty results. This is a composite score, not ordinary recall. |
| `conflict_pair_retrieved` | Both conflicting sources were returned; neither is declared correct. |
| `conflict_pair_linked` | Returned evidence includes an explicit conflict relation between the labelled pair. |
| `lexical_candidate_count` | Count offered by the engine's lexical candidate generation, before its shortlist cap. |
| `ranking_candidates_offered` | Actual number of relevance questions passed to JEV, observed at the provider boundary. |
| `quality_metrics_valid` | Whether this row can contribute to the stated mode's aggregate. A failed JEV call followed by lexical fallback cannot. |
| `live_ranking_performed` | Whether JEV successfully reranked this query. Zero-candidate outcomes need no remote ranking call. |
| `telemetry` | Provider counters, including requests, retries, questions, cache hits, and validated token usage. |

Source recall uses unique source keys among at most `k` returned claims. The
fixtures currently contain one paragraph per source. A larger corpus with several
claims per source would need a separate source-level retrieval budget.

The no-overlap synonym query deliberately exposes the current retrieval boundary:
“Do cars consume gasoline?” cannot retrieve “Automobiles burn petrol.” through
lexical candidates alone. JEV cannot improve evidence it was never offered. The
other synonym query retains a lexical anchor, so reranking can help distinguish
relevant evidence from a distractor. Unanswerable queries cover both a misleading
shared project name and zero lexical overlap.

Timing includes local work, service latency, and retries. Validated response token
counts may omit failed responses and are not a billing record. The report also
records corpus and implementation hashes so results can be tied to exact inputs
and code. Run multiple clean trials before discussing latency variability.

## Limits

No live scores are checked into this package. Structural tests of the adapter,
citations, and lifecycle do not establish semantic quality. This tiny corpus can
expose regressions and candidate-generation limits; it cannot establish superiority
over Cognee, mem0, a vector index, or a generative memory pipeline. It does not test
answer generation, real user histories, harness injection, sustained concurrency,
or long-term forgetting. Those need separately labelled, representative datasets.

First live results: [live-evaluation-2026-09-27.md](live-evaluation-2026-09-27.md).
LongMemEval intake results: [intake-sentence-claims-2026-09-27.md](intake-sentence-claims-2026-09-27.md).
Live sweep on main with a finer keep-threshold sweep: [longmemeval-live-2026-09-28.md](longmemeval-live-2026-09-28.md).
