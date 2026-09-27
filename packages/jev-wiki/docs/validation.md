# Build validation

## PR #1 review — 27 September 2026

Reviewed PR head `12ae10be9d4ab5e7e77ce73f91473d79e4ed5785`. The recovered
source archive SHA-256 was
`2606fae4bd6d5b99c1d20d382ed0498fa21e2e74c0f2995528eca60f53d3c07c`.
Every package blob/subtree matched that GitHub revision before changes.

Environment: Linux, Python 3.13.5. All 86 original tests passed before review.
Ten initial reproductions failed against the original implementation; the final
suite adds 20 regression tests and passes all 106 tests. See
[the review findings](pr-1-review.md) for the fixes and remaining acceptance gates.

| Check | Observed result on the reviewed changes |
| --- | --- |
| `PYTHONPATH=packages/jev-wiki python3 -m unittest discover -s packages/jev-wiki/tests -q` | 106 tests passed |
| `python3 -m pip install --no-index --no-deps --no-build-isolation -e ./packages/jev-wiki` | Passed in a local virtual environment using preinstalled build tools; no packages downloaded |
| Installed environment: `python3 -m unittest discover -s packages/jev-wiki/tests -v` | 106 tests passed, including CLI subprocess tests |
| `python3 packages/jev-wiki/examples/demo.py --root <new-directory>` | Passed conflict, revision, forgetting and exact-citation assertions; scripted provider only |
| `jev-wiki --root <demo-directory> --provider none lint` | `ok: true`, no issues |
| `python3 -m compileall -q packages/jev-wiki` | Passed |
| `git diff --check` | Passed |
| Ruff lint and format | Not rerun: Ruff is not installed in this environment; earlier results below apply only to the original build |
| Live JEV / actual Claude Code session | Not run: no JEV key or Claude Code host available |

No paid CI was requested. These results establish the exercised local software
paths, not live semantic quality, pricing, latency, or production readiness.

## Original build — 24 September 2026

Environment: Linux, Python 3.12.14. The package was installed with
`python3 -m pip install --no-deps -e ./packages/jev-wiki`; it does not install Cognee.

| Check | Observed result |
| --- | --- |
| `python3 -m unittest discover -s packages/jev-wiki/tests -q` | 86 tests passed |
| `python3 -m ruff check packages/jev-wiki` | Passed using the repository's rules |
| `python3 -m ruff format --check packages/jev-wiki` | Passed |
| Installed CLI subprocess lifecycle | Init, deferred capture, process, worker, offline recall, lint and forgetting passed |
| `python3 examples/demo.py --root <new-directory>` | Completed; conflict, revision, forgetting and lint assertions passed |
| Live evaluation without credentials | Refused with exit 2; no network call or invented results |

The test suite covers exact Unicode citations, immutable raw hashes, retries,
malformed API responses, independent batched decisions, reordered answer maps,
multiple writer processes, source revision/deletion during inference, and late
workers preserving completed human review. Hook tests exercise real event shapes
through subprocesses and a local deadline; no actual Claude Code host was installed.

The scripted demo creates two incompatible launch-colour sources and retains both
with conflict annotations. A preference revision replaces the earlier current
evidence. Forgetting one colour source removes that evidence and its dangling
conflict annotation. Every returned passage retains its raw revision and offsets.

**Not measured:** live JEV semantic accuracy, threshold calibration, latency, cost,
or performance against Cognee/Mem0. `TYPESAFE_API_KEY` was unavailable. The live
[evaluation harness](evaluation.md) is provided for that next gate. These checks
qualify the software paths; they do not establish model quality or production scale.
