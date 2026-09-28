# Continuation

The executable first version lives entirely in `packages/jev-wiki`. Cognee remains
available as a comparison baseline; the package does not depend on its runtime.

## First gate: measure live JEV

Supply `TYPESAFE_API_KEY` and run the documented live evaluation. Review every missed
memory, irrelevant inclusion and conflict label before changing confidence thresholds.
Record the actual model, corpus revision, latency and token usage. The scripted demo
and software tests are not evidence of semantic performance.

## Next improvements, driven by that evidence

1. If relevant sources miss the lexical shortlist, add a bounded candidate-expansion
   step (JEV topic routing or optional embeddings), then compare against the same queries.
   Partly done offline: BM25 ([bm25-shortlist-2026-09-27.md](bm25-shortlist-2026-09-27.md))
   a source-context lift
   ([source-context-shortlist-2026-09-27.md](source-context-shortlist-2026-09-27.md)),
   and optional local embedding candidates
   ([embedding-candidates-2026-09-27.md](embedding-candidates-2026-09-27.md)). JEV topic
   routing is untested: it needs live intake labels. Rerun the live sweep with the
   embedder on once credits return.
2. Done in rubric `wiki-v2`: sentence candidates with neighbouring context and exact
   offsets, plus keep wording that values personal facts. See
   [intake-sentence-claims-2026-09-27.md](intake-sentence-claims-2026-09-27.md). Rubric
   `wiki-v3` files a request by the personal facts it states; see
   [intake-requests-2026-09-28.md](intake-requests-2026-09-28.md).
3. If richer narrative pages materially help recall, add an optional writer that
   proposes cited prose. Validate each supporting span; synthesized answers must
   never become independent corroborating sources.
4. If the small-wiki state limit or read latency matters, retain portable raw/JSON
   ownership and add a rebuildable SQLite index and incremental transaction journal.

## Deployment qualification

Install the sample hooks into one test project, manually merging settings. Verify
capture, worker recovery, prompt injection and failure-open behavior in a real
Claude Code session. Hook subprocess tests are provided; no live host was available
in the build environment. Add other harness adapters only after checking their
actual supported event contracts.

Do not broaden automatic semantic supersession or deletion based on model confidence
alone. Do not treat cheap repeated judgments as independent evidence. Add a purge
operation only with explicit raw/cache/audit/backup retention semantics.
