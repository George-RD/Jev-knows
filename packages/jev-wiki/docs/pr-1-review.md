# PR #1 review — 27 September 2026

Base reviewed: `12ae10be9d4ab5e7e77ce73f91473d79e4ed5785` on
`feat/wiki-first-jev-memory`. No inline review threads existed when this review
started. This is an implementation review with local reproductions, not a live
model evaluation. Only the standalone `packages/jev-wiki` package is changed.

## Findings addressed

### Project isolation and event provenance — high priority

`drain_inbox` checked event hashes but did not compare the event's project with
the root's saved binding. A correctly hashed event copied from another project
was ingested. It also accepted `Stop` events with a `user` role. Hash integrity
alone did not enforce the capture contract.

The worker now verifies the binding, schema and hashed identifiers, exact
`UserPromptSubmit/user` or `Stop/assistant` pair, nonempty text and 24 KB message
limit before ingestion. Invalid, foreign and unbound events remain available for
inspection and count as failed. Bounded nonblocking regular-file reads also
protect project-binding and event reads. The filesystem remains a trusted local
boundary; this does not authenticate an event against a malicious local writer.

### Hook evidence was unnecessarily omitted — medium priority

The hook divided its remaining 6,000-character packet budget by five before
recall, even for ordinary text. A complete, relevant paragraph of about 1,400
characters was dropped although its quote, citation and wrapper easily fit.

Recall now accepts a trusted output-size measure. The hook supplies actual
HTML-escaped length, reserving the fixed wrapper separately. It preserves whole
quotations and citations, skips oversized evidence without truncation, and can
still include smaller relevant evidence. The direct recall default is unchanged.

### Response-cache privacy and blocking reads — high priority

The provider validated known fields but cached the complete response object.
Unexpected debug fields could therefore persist echoed source text or headers.
A FIFO at the cache path could also block a worker outside its HTTP timeout.

Only validated model, answer and token-usage fields are now written. Existing
valid entries with extra fields are sanitized when accessed; untouched historical
cache files are not bulk-erased. Cache reads reject special files and symlinks,
use nonblocking descriptors and enforce byte bounds. A rejected optional cache
falls back to the provider. Symlink targets are not read or changed.

### HTTP and numeric failures escaped validation — medium priority

Truncated HTTP bodies raised `http.client.IncompleteRead` outside the retry path.
Extremely large numeric answers overflowed during conversion before validation,
escaping the intended `ProviderError` boundary.

HTTP protocol exceptions now use the existing three-attempt retry policy and
redacted terminal error. Numeric bounds are checked before float conversion in
the adapter, engine and claim validation; invalid confidence is not promoted.

### Escaping could permanently defer valid sources — medium priority

Processing always grouped four paragraphs. Raw text could fit source limits but
exceed the adapter's state/question budget after nested JSON escaping, causing
every retry to defer the same source before making a request.

Processing now uses up to four paragraphs while also measuring the state as
encoded inside the request. It preserves global candidate identifiers, exact
quotes and character offsets. The real adapter still enforces its complete
state/question and request bounds; the new transport test exercises those checks
without a live request. This does not remove lexical recall limitations.

### Concurrent forgetting could abort unrelated pending work — medium priority

A pending-source snapshot could become stale while processing an earlier source.
Attempting the now-forgotten source raised `ValueError` and stopped later work.
Forgetting during offline or failed-provider deferral could likewise escape.

The batch worker now marks retracted/replaced sources as cancelled and continues.
Deferral handles concurrent completion and retraction without overwriting a
completed result. Validation errors for a still-current source still propagate;
they are not silently treated as cancellations.

## Verification and remaining gates

All 106 tests pass, including 20 new regressions in
`tests/test_review_regressions.py`. The initial ten reproductions were run against
the original implementation before fixes. Editable installation, the scripted
lifecycle demo, generated-wiki integrity checks, compilation and diff whitespace
checks also pass. Exact commands and environment are in [validation.md](validation.md).
Ruff was unavailable and was not rerun; earlier Ruff results do not qualify this
new patch.

Keep the PR draft pending real JEV evaluation with `TYPESAFE_API_KEY` and
acceptance in an actual Claude Code session. Neither was available here. No
semantic accuracy, latency, cost or Cognee/Mem0 parity result is claimed. No
existing Cognee runtime, MCP server or frontend is changed. Forgetting remains
logical retraction, not secure erasure.
