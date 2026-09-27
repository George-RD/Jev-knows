# Jev Wiki

An opinionated Karpathy-style memory wiki for agents. Original sources stay intact.
JEV selects useful passages, classifies them, reranks recall, and checks possible
conflicts. Code compiles the decisions into cited Markdown and maintains source
revisions. Agents can capture and recall through hooks without choosing a memory tool.

This is a standalone package inside Jev Knows. It has **no runtime dependencies**
and does not start or import Cognee. Python 3.10+ on Linux or macOS is required;
the file lock uses POSIX `flock`.

## Start

From this repository's root:

```bash
python3 -m pip install -e ./packages/jev-wiki
jev-wiki --root /absolute/path/to/my-memory init
```

Set `TYPESAFE_API_KEY` through your shell or secret manager, then:

```bash
jev-wiki --root /absolute/path/to/my-memory ingest notes.md --source-key project:notes
jev-wiki --root /absolute/path/to/my-memory recall "What did we decide?"
jev-wiki --root /absolute/path/to/my-memory maintain --max-pairs 20
jev-wiki --root /absolute/path/to/my-memory lint
```

Without a key, ingestion retains the source and reports `deferred`. It does not
invent model results. Add a key and run `process` to retry pending sources. Use
`--provider none` for explicitly local operations and `recall --offline` for lexical
retrieval over already accepted memories. CLI provider selection defaults to `auto`:
if a key is present, processing and ordinary recall send bounded evidence to TypeSafe.

Recall can also use a small local embedding model, so a question about "dinner with my
homegrown ingredients" can find a note that only mentions basil. It is off by default:

```bash
pip install "jev-wiki[embed]"
export JEV_WIKI_EMBEDDING_MODEL=default   # or a model2vec model name or local path
```

The first use downloads `minishlab/potion-retrieval-32M` (129 MB) from Hugging Face;
a local path needs no network. `recall` then adds the claims most similar to the query
to its shortlist. The embedding itself runs on your machine; with `TYPESAFE_API_KEY`
set, ordinary `recall` still sends the query and shortlisted claims to TypeSafe for
reranking. `recall --offline` stays lexical.

The prompt hook uses the same model when it is already downloaded: it never downloads,
reads the model memory-mapped (about 0.1 s), and reads claim vectors that `worker` and
`recall` save under `<root>/embeddings/`. If loading or encoding new claims would run
past 0.55 s of the hook's 0.75 s budget, that prompt's recall stays lexical. See
[docs/hook-embeddings-2026-09-27.md](docs/hook-embeddings-2026-09-27.md).

Re-ingest a changed document with the **same source key** to replace its current
revision. Independent sources need independent keys: capture order does not resolve
their factual disagreements.

## What is opinionated

| Decision | Result |
| --- | --- |
| Evidence first | Each memory is an exact source passage with revision and character offsets |
| JEV decides; code writes | Independent questions share bounded state; JEV never chooses file paths or writes prose |
| Conservative promotion | Uncertain passages and assistant/tool output enter the review queue, outside ordinary recall |
| Conflicts remain visible | Maintenance annotates both assertions; it never chooses a winner or erases a fact |
| Readable, rebuildable wiki | Topic pages, review queue, index and log are generated from raw files and JSON state |
| Explicit project boundary | Hooks require both a memory root and project root |
| Background processing | Hooks capture locally; a separate worker makes model calls and runs bounded maintenance |

JEV is used for three independent decisions per source passage: retention, kind and
topic. Recall uses relevance scores. Maintenance evaluates shortlisted pairs for
conflict, duplication or relatedness. The adapter supports all three JEV primitives,
pins `jev-1.13.0`, validates distributions, caches versioned results, and records actual
usage. State and question batches have conservative byte limits.

The retention and confidence thresholds are initial policy choices, **not calibrated
accuracy guarantees**. Exact citations establish what a source said, not whether it
is true. Prose synthesis and open-ended entity extraction are future extensions.

## Automatic agent memory

[Claude Code setup](docs/hooks.md) provides `UserPromptSubmit` and `Stop` hooks.
Prompt hooks inject bounded local evidence; Stop captures the final assistant message
as review material. Neither reads arbitrary transcript files or calls JEV.

```bash
jev-wiki --root /absolute/path/to/my-memory worker --limit 100
```

Run the worker yourself or schedule that command on the host. No hook configuration,
background service or scheduled job is installed automatically. Direct CLI recall can
use JEV reranking; prompt hooks never call a provider, and use local embedding
candidates only when a downloaded model loads within their budget.

## Files and editing

- `raw/`: immutable UTF-8 source revisions.
- `state.json`: canonical sources, exact spans, decisions, relationships, processing
  state, tombstones and audit events; changes use a process lock and atomic replacement.
- `wiki/generated/`: generated topic pages and review queue. Rebuildable; do not edit.
- `wiki/index.md`, `wiki/log.md`: navigation and activity. Existing handwritten
  versions are preserved; a generated index is always available separately.
- `wiki/notes/`: a suggested home for handwritten notes. Ingest them explicitly under
  stable source keys to make them available to recall.
- `AGENTS.md`: local wiki instructions created by `init`, without replacing existing instructions.
- `inbox/`: durable hook events and receipts; `cache/jev/`: optional model response cache.

Rebuild pages without inference:

```python
from jev_wiki.store import WikiStore

WikiStore("/absolute/path/to/my-memory").render()
```

Review is explicit: inspect a claim and its source, then use
`WikiStore(...).update_claim(claim_id, {'status': 'active'})` when accepting it is
warranted. This checks the source span again and records the change. No automated
rule promotes assistant output after merely repeating it.

`forget --source-key KEY` retracts all revisions from current recall and generated
pages and prevents accidental recapture of that key. **It is not secure erasure**:
raw revisions and audit state remain. Keep personal memory roots outside public
repositories; Git backup is optional and never pushed by this package.

## Test and evaluate

```bash
cd packages/jev-wiki
python3 -m unittest discover -s tests -v
python3 examples/demo.py --root /tmp/jev-wiki-demo
```

The demo uses visibly scripted decisions to exercise conflicts, source replacement,
forgetting, citations and reconstruction. It is not a JEV benchmark.

[Build validation](docs/validation.md): the PR #1 review passes 106 tests,
editable installation and the executable demo. The original 86-test build also
passed Ruff; Ruff was not available for the review. See the
[review findings](docs/pr-1-review.md) for fixes and remaining live checks.

[The live evaluation](docs/evaluation.md) compares lexical recall with JEV reranking
over a small synthetic corpus, with held-out queries and token/latency reporting.
It requires a real key and makes real requests. No live semantic results were
measured during this implementation because a key was unavailable.

## Current limits

This initial implementation targets small, single-user wikis. BM25 shortlisting
can miss synonyms before JEV sees a candidate; the optional embedding extra (below)
narrows that gap. Maintenance checks a bounded selection
of pairs, not every possible contradiction. Claims are source sentences: a sentence
holding several assertions stays quoted whole, and sentence splitting is a heuristic
(abbreviations it does not know can split a sentence early). There is no generative summarizer, automatic
entity graph, vector database, distributed service, or Cognee benchmark parity claim.

Sources are limited to 200 KB per ingest, query length to 2,000 characters, and
canonical state to 16 MiB. A full state file is read for operations; this is not an
unbounded archive. Remote TypeSafe inference, live model quality and an installed
Claude Code session require deployment validation.

See [architecture](docs/architecture.md), [verified upstream sources](docs/sources.md),
and [continuation work](docs/continuation.md).
