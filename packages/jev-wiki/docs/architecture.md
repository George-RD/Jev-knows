# Jev Wiki architecture

This package implements an opinionated, small memory system inside Jev Knows. It keeps
source evidence and a readable Markdown wiki together. JEV makes bounded semantic
decisions; ordinary code owns persistence, exact quotes, lifecycle transitions and
rendering. The initial package uses Python's standard library and does not require
Cognee's graph, vector or LLM services.

This document describes the package's design contract. Tests establish the behavior
actually exercised. Offline and scripted-provider tests establish software behavior,
not the accuracy, calibration, speed or cost of the live JEV model.

## Foundation and deliberate choices

[Karpathy's LLM Wiki pattern](sources.md#karpathy-llm-wiki) supplies the separation of
raw evidence, compiled Markdown and maintainer instructions, plus ingest, query and
lint operations. It does not prescribe this package's schema, APIs or automatic
capture policy.

Our choices are:

- Keep the first installation local and inspectable: immutable raw revisions,
  canonical JSON state, generated Markdown and lexical candidate search in memory.
- Select and classify source paragraphs instead of asking JEV to invent claims,
  names or prose. Copy accepted text directly from source evidence.
- Use the same narrow provider contract for filtering, ranking and pair
  relationships: `Provider.ask(state, questions) -> answers`.
- Make capture automatic where a harness supports it; let an explicit worker process
  queued captures and make remote requests. A hook firing is not itself consent to
  upload every transcript or local file.
- Record uncertainty and contradictory evidence. A model's confident answer does
  not make source material authoritative.

## Data ownership

| Layer | Owns | Reconstruction and editing |
| --- | --- | --- |
| Raw revisions | Original text, stable source identity and content digest | Evidence is retained unchanged during ordinary updates; explicit forgetting has different retention semantics |
| Canonical JSON state | Source metadata, processing status, selected spans, decisions, relationships and tombstones | This is the committed application state; raw text alone cannot reproduce past probabilistic decisions |
| Hook inbox | Pending capture events and content-free processed-event receipts | Separate files under `inbox/`; the worker captures each event before consuming it |
| Generated Markdown | Classified memory pages, review queue, raw-source links, index and activity log | Rendered from canonical state; direct edits are not imported into state |
| Retrieval candidates | Lexical matches over active memory | Computed from current state; no separate vector or SQLite database |
| Provider adapter | Request construction, response validation and remote boundary | Receives bounded data; cannot directly edit sources or pages |
| Harness adapter | Event capture and bounded context output | Uses the same memory engine; does not establish a second memory store |

A source key identifies one logical source. A content digest identifies a revision.
Re-ingesting the same key and content is idempotent. New content creates a revision;
it must not silently change the evidence underlying a previously selected span.
Every displayed memory must retain a link to its source revision.

## Engine lifecycle

The engine surface is deliberately small:

```python
ingest(text, source_key, title="", metadata=None)
process(source_id)
recall(query, limit=5, max_chars=6000, offline=False)
maintain(max_pairs=20)
forget(source_key)
```

`ingest` captures evidence and calls `process`; without a provider it records deferred
work. Hooks have a separate inbox so their capture path does not perform remote
processing. `process` constructs paragraph candidates and asks bounded questions
about their usefulness and classification. Code applies
validated decisions and renders the resulting pages. A provider failure must leave
raw evidence available and work recoverable; it must not create successful-looking
model output.

`recall` makes a lexical shortlist, optionally asks JEV to judge relevance, and emits
a bounded context pack with citations. The shortlist is a real recall limitation:
reranking cannot recover a passage never offered as a candidate. Unmatched queries
must not be padded with unrelated material merely to reach `limit`.

`maintain` spends a bounded pair budget investigating potentially related memories.
Code prioritizes pairs by lexical overlap; JEV classifies their relationships as
`conflict`, `duplicate`, `related`, `unrelated` or `uncertain`.
Those relationships are annotations, not authorization to erase evidence.
The initial relation pass is not a complete semantic audit of all possible pairs.

`forget` records a permanent source-key tombstone, excludes all its revisions from
active memory, cancels its hook inbox entry and regenerates affected views. Raw text,
canonical claims and event history remain locally for audit. The same key cannot be
captured again; new material requires a new source key. This command is logical
forgetting, not physical erasure. It does not remove prior backups, Git history,
host transcripts or content already sent to an external inference system.

## Making useful use of JEV

JEV's [documented primitives](sources.md#typesafe-primitives-and-api) fit decisions
over supplied candidates. They do not provide arbitrary free-text generation.

| Stage | JEV judgment | Code responsibility |
| --- | --- | --- |
| Processing | Is this paragraph durable and useful? Which known category fits? | Paragraph boundaries, IDs, exact text and citations |
| Recall | Does each shortlisted memory help answer this query? | Candidate generation, ordering policy and output budget |
| Maintenance | How are these two passages related? | Lexical pair priority, pair budget, lifecycle rules and preservation of evidence |
| Failure handling | None | Timeouts, errors, retries, local fallback and honest status |

Batch independent questions that share relevant state. Do not assemble the entire
wiki merely because the API permits a large request. Use explicit unknown/none
options where a closed choice otherwise forces a false match. Keep arithmetic,
date ordering and identity checks in code. Record model identity and decision
provenance sufficiently to distinguish live decisions from offline demonstrations.

[TypeSafe's confidence field](sources.md#confidence) summarizes an answer's
probability distribution. It is not a separate source-verification result. Thresholds
are initial policy settings until a held-out evaluation establishes suitable values
for this corpus and model version. Rephrasing a question repeatedly does not produce
independent evidence.

## Correctness boundaries

### Revisions, time and contradictions

Observation time answers when the system saw evidence, not when the evidence became
true. Publication time, event time and validity intervals are different concepts.
The first version does not infer a complete temporal model from arbitrary prose.
Do not interpret the latest ingested passage as the winner of a factual dispute.

Replacing a source should stop old-revision material being presented as current.
Independent conflicting sources should remain visible as a conflict. Automatic
semantic supersession and claim-level temporal reasoning require additional
evaluation and are outside the initial package.

### Concurrent writes and recovery

Writers use the store's lock and atomic state replacement. Provider requests should
not hold a write lock for their network duration. A result must still refer to the
source revision it evaluated when applied: if the source was replaced or forgotten,
discard the obsolete result. Canonical state is the recovery authority if rendering
is interrupted; rebuilding views must not require another model call.

This is a single-machine POSIX filesystem design using `flock`. It is not a
distributed transaction protocol and does not make network filesystems or concurrent
Git merges safe. Windows support is not established.

### Untrusted content

Documents, chat messages and retrieved memories are data. They cannot define
provider URLs, application settings, command arguments, file paths or schema rules.
Generated pages should show excerpts as quoted evidence, retaining provenance.
Only trusted configuration controls the remote destination and local store.

JEV itself can be influenced by adversarial state, according to its
[documented limitations](sources.md#known-model-limitations). A classifier saying a
passage is safe is not a security guarantee. Output validation and restricted write
surfaces still matter. A downstream agent must continue treating injected memory as
quoted context rather than higher-priority instructions.

### Automatic capture and context injection

Capture should preserve whether content came from a user, assistant or tool. An
assistant's unverified output is not independent factual evidence. Scope capture to
the configured project and supported event fields; do not silently read arbitrary
paths carried in event payloads.

The initial hook path queues locally. Remote processing is an explicit worker step.
Recall context has a size budget and an identifiable evidence wrapper. Replayed
hooks must not multiply captures. A failed memory hook must not trap the main agent
in a stop/retry loop.

## Non-goals of this version

- Free-form multi-document prose synthesis or an autonomous research agent.
- Guaranteed contradiction discovery, truth verification or adversarial immunity.
- Graph/vector database infrastructure, embeddings or a distributed service.
- Automatic replacement of Cognee's existing public APIs.
- Multi-user permissions, tenant isolation, encrypted storage or compliance claims.
- Installation into every agent harness, or claims of live harness testing without
  an actual installed host.
- Claimed parity with Cognee, Mem0 or another memory benchmark without comparable
  measured runs.

## Evidence required before stronger claims

Lifecycle tests should cover source revision changes, exact citations, retryable
provider failure, concurrency, forgetting, reconstruction and bounded recall. Hook
tests should exercise real event payload shapes with isolated temporary stores.

A separate live evaluation needs held-out labeled captures and queries. Compare
lexical retrieval with JEV reranking, report missed relevant memories and false
promotions, and measure actual request latency, input usage and cost. Record model
version, question version and corpus revision. Scripted answers can test branches;
they cannot supply that semantic evidence.
