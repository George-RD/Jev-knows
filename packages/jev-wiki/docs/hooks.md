# Claude Code integration

The adapter captures prompts and completed assistant messages into a local
inbox. Before each prompt, it retrieves already compiled memory using a bounded
local search: BM25, plus embedding candidates when `JEV_WIKI_EMBEDDING_MODEL` names a
model that is already downloaded and loads within the hook's budget. Only the separate worker sends source text to JEV. Capture and
recall therefore do not depend on the main assistant choosing a memory tool.

The [official hooks reference](https://code.claude.com/docs/en/hooks) was checked
on 24 September 2026. The adapter uses `UserPromptSubmit.prompt`,
`Stop.last_assistant_message`, common `session_id` and `cwd` fields, and
`hookSpecificOutput.additionalContext`. It never reads `transcript_path`.
This integration has synthetic protocol tests; a live Claude Code session is
still a separate acceptance check.

## Install in one project

From the Jev-knows repository root, create a small, separate environment:

```sh
python3 -m venv .venv-jev-wiki
.venv-jev-wiki/bin/python -m pip install ./packages/jev-wiki
.venv-jev-wiki/bin/python -m jev_wiki --root "$PWD/.jev-wiki" init
```

These commands install only the standalone package. For another project, create
the environment in that project's root and replace `./packages/jev-wiki` with
the absolute path to this package. The store currently requires POSIX file
locking; use Linux, macOS, or WSL rather than native Windows.

Manually merge the `hooks` entries from
[`examples/claude-settings.json`](../examples/claude-settings.json) into the
project's `.claude/settings.local.json`, preserving existing hooks. The example
uses the exact `.venv-jev-wiki/bin/python` path created above. It uses
`CLAUDE_PROJECT_DIR` for both paths and project scope, so a later `cd` does not
silently select another memory. The sample sets a two-second host timeout.
Do not put this project-specific example into global settings.

No command here automatically changes Claude settings or installs a scheduler.
Add `.jev-wiki/` and `.venv-jev-wiki/` to your project's local Git exclusions;
captured conversations are private project data.

## Compile and maintain

Supply `TYPESAFE_API_KEY` in the worker's environment using your normal secret
management, then run:

```sh
.venv-jev-wiki/bin/python -m jev_wiki --root "$PWD/.jev-wiki" worker --limit 100 --max-pairs 20
```

Each run captures up to 100 queued events, attempts up to 100 pending or deferred
sources, and checks at most 20 relationship pairs. `--no-maintain` skips that last
stage. Re-run this command after a session or invoke it with your existing local
scheduler. It is a single pass, not a daemon. Without the API key, sources stay
deferred and are retried by a later run. CLI `--provider none` forces local-only
operation even if a key exists.

```sh
.venv-jev-wiki/bin/python -m jev_wiki --root "$PWD/.jev-wiki" recall "What decisions did we make?" --offline
.venv-jev-wiki/bin/python -m jev_wiki --root "$PWD/.jev-wiki" lint
```

The worker preserves a message's role. Assistant output remains a review
proposal and is not promoted to independent evidence. Handwritten notes under
`wiki/notes/` enter compilation only when explicitly ingested:

```sh
.venv-jev-wiki/bin/python -m jev_wiki --root "$PWD/.jev-wiki" ingest .jev-wiki/wiki/notes/decision.md --source-key wiki:decision --title "Project decision"
```

## Scope, failure behavior, and limits

- A memory root must be explicit. The first hook binds it to the configured
  project. Another project or a `cwd` outside that project receives no memory.
  Subdirectories share project memory; separate sessions have separate capture
  identities but retrieve the same project's compiled evidence.
- Hook payloads are limited to 64 KiB; captured message text to 24,000 UTF-8
  bytes. Oversized or malformed events are ignored, with no model call. For a
  long valid prompt, the retrieval query uses its first and last 1,000 characters.
- Returned context is at most 6,000 characters, enclosed in an explicit
  untrusted-evidence fence. Escaping cannot close that fence. Whole evidence
  blocks are omitted when they cannot fit; quotations are never clipped.
- Missing memory, missing fields, unsupported events, provider outages, malformed
  input, and local storage errors never block the user's prompt. The hook emits
  only JSON, exits zero, and supplies no permission or continuation decision.
  A 750 ms local wall-clock budget also interrupts blocked reads or store locks.
  The two-second host timeout is an additional bound and can discard late output.
- `inbox/hooks/` contains pending message text. Receipts contain only an event
  hash and state. A stable hash of project, session, event, role, and text makes
  retries idempotent, including across worker crashes. Identical repeated text
  in the same session is deliberately collapsed. A moved project gets a new
  scope identity; explicitly choose a new memory root or reconfigure its binding.
- The inbox uses atomic publication. A worker can replay after an interruption;
  store source-key idempotency prevents duplicate evidence. This is at-least-once
  processing, not a guarantee that concurrent workers make only one JEV request.
- Forgetting a hook source cancels its queued event and keeps a content-free
  receipt. Store deletion tombstones prevent the same key being silently revived.
  Remove sources from retrieval through the CLI instead of editing the inbox or
  state. Immutable raw audit bytes remain; this is not physical data erasure.

## Acceptance check

After installing the sample in a live project, submit a harmless factual prompt,
run the worker, then ask about that fact in a fresh session. Check the memory's
source citation and Claude's hook debug log. Stop capture requires a Claude Code
version that supplies `last_assistant_message`; older versions simply capture
prompts. No complete conversation or tool-output capture is claimed.

## Worker validation and evidence budget

The worker rechecks the saved project binding before ingesting each queued event.
It validates the schema, hashed identifiers, event/role pair, nonempty text and
message byte limit, in addition to the content hash and source key. An event from
another project, an assistant message labelled as a user prompt, or an event in
an unbound root remains queued for inspection and increments `failed`; it is not
ingested. The binding is local scoping, not authentication against someone able
to rewrite the memory directory.

Recall measures the actual HTML-escaped packet size before selecting each whole
quotation and citation. Plain text can use the available budget without reserving
a fivefold expansion for every character. Escaping-heavy quotes that cannot fit
are skipped, not cut short; other complete, relevant evidence can still fit.
Project-binding and queued-event reads reject special files and symlinks and are
bounded by bytes, so a FIFO cannot stall the worker while it reads an event.
