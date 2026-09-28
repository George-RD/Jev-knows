# Hook store read (2026-09-28)

Before this change the prompt hook returned no memory on wikis above about 6,000
claims. `Engine.recall` read every claim through `WikiStore.claims()`, which re-reads
and hashes every raw source, round-trips every claim through JSON to revalidate it,
and deep-copies the result. It then parsed `state.json` a second time in
`active_evidence` to recheck the claims it emits. On a 6,000-claim wiki that is
about 0.45 s before any ranking, against a 0.75 s hook budget.

## Change

- `WikiStore.recall_snapshot()` returns current sources and active claims (with
  relations filtered to active targets, as `claims()` does) from one locked read, with
  no raw-source reads, per-claim revalidation, or deep copies. `Engine.recall` ranks
  from it.
- Fail-closed checking moved entirely to the claims recall emits: `active_evidence`
  already revalidated each emitted claim's current status, exact quote and raw-source
  hash under the lock after ranking, and it still does. A tampered claim that recall
  would emit still raises; tampered evidence that is not emitted no longer blocks
  recall of other evidence. `claims()` itself keeps full validation for every other
  caller, such as maintenance and worker indexing.
- `recall_snapshot` and `active_evidence` share one parse of `state.json` while its
  identity (device, inode, size, mtime, ctime) is unchanged. Every store write
  replaces the file, so a write always invalidates it. At 10,000 claims each parse
  costs about 0.1 s. Editing `state.json` in place by hand is unsupported, as before.

Ranking is unchanged: the snapshot holds the same claims in the same order with the
same fields as `claims()` (a unit test compares them), so the shortlist and the
emitted items are identical.

## Timing

Wikis built from LongMemEval_S haystacks (the first 24 questions' sessions, every
sentence kept and active, as in the offline scorer). `python -m jev_wiki hook` in a
fresh process, median of 7 prompts, wall time including interpreter start-up. With
embeddings, the worker had indexed every claim first.

| Claims | state.json | Lexical before | Lexical after | Embeddings before | Embeddings after |
|---|---|---|---|---|---|
| 3,479 | 3.9 MB | 0.44 s | 0.18 s | 0.71 s (5/7 answered) | 0.38 s |
| 6,196 | 6.9 MB | 0.77 s | 0.27 s | 0.81 s (4/7 answered) | 0.42 s |
| 10,059 | 11.3 MB | 0.81 s (0/7 answered) | 0.41 s | 0.81 s (0/7 answered) | 0.55 s |

"Answered" counts prompts that got memory before the 0.75 s alarm; after the change
every prompt at every size did. In-process, recall at 10,059 claims went from 1.02 s
to 0.28 s; the embedded hook recall stayed hybrid at every size.

## Recall quality

[`examples/longmemeval_offline.py`](../examples/longmemeval_offline.py), all 470
questions:

| | @1 | @3 | @5 | @10 | recall_all@10 | Preference @10 |
|---|---|---|---|---|---|---|
| Lexical, before | 0.832 | 0.909 | 0.940 | 0.962 | 0.864 | 0.800 |
| Lexical, after | 0.832 | 0.909 | 0.940 | 0.962 | 0.864 | 0.800 |
| Embeddings, before | 0.866 | 0.949 | 0.966 | 0.983 | 0.936 | 0.967 |
| Embeddings, after | 0.866 | 0.949 | 0.966 | 0.983 | 0.936 | 0.967 |

"Before" is the PR #9 merge (`6ca218e5`), scored in the same environment; every
question type matched too. (PR #9's doc reports @3 0.951 with embeddings; that
commit scores 0.949 here, identically before and after this change.)

## Limits

- At 10,000 claims the remaining cost is one 11 MB JSON parse (about 0.1 s),
  tokenizing every claim for BM25 (about 0.14 s), and hashing every claim for its
  vector. The hook still reads the whole wiki on every prompt.
- `state.json` has a 16 MiB safety limit, which LongMemEval-sized claims reach at
  about 14,000 claims. That, not hook time, now bounds wiki size.
