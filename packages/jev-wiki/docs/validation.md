# Build validation — 24 September 2026

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
