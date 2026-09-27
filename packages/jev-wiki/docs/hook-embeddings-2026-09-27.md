# Embedding candidates in the prompt hook (2026-09-27)

[PR #8](embedding-candidates-2026-09-27.md) added local embedding candidates to CLI
`recall`, but the prompt hook, which is where agents actually recall, stayed lexical:
loading the model through model2vec took about 0.8 s against the hook's 0.75 s budget,
and every recall re-encoded every claim. This change lets the hook use the same
candidates within its budget. Measured without JEV (the TypeSafe account was out of
credits), on Linux with 4 CPU cores while the offline scorer was also running, so the
times below are pessimistic.

## Where the load time went

| Step (potion-retrieval-32M) | model2vec `from_pretrained` | Memory-mapped loader |
|---|---|---|
| numpy + tokenizers import | 0.16 s (model2vec, huggingface_hub, ...) | 0.07 s |
| Model files | 0.6 s: Hugging Face checks, then reading the 129 MB table | 0.001 s: `np.memmap` of the table |
| Tokenizer | included above | 0.06 s |
| First sentence | 0.001 s | 0.001 s |
| **Total, in a fresh process** | **0.75–1.2 s** | **0.13–0.21 s** |

A smaller model does not help: `potion-base-8M` took the same time with either loader
(0.75 s and 0.17 s), because its cost is the imports and the shared `bge` tokenizer,
not the table. It also finds five fewer questions than the 32M model on its own
(PR #8's table), so the default stays `potion-retrieval-32M`.

`StaticEmbedder` now reads `model.safetensors` and `tokenizer.json` directly: the
token-vector table is memory-mapped, so a sentence reads only the rows of its own
tokens. Encoding follows model2vec's `StaticModel.encode` (no special tokens, unknown
tokens dropped, 512-token cut, mean of weighted rows). On 3,208 LongMemEval texts the
vectors matched model2vec's to 6e-8 for both `potion-retrieval-32M` and
`potion-base-8M`. Float16 tables, per-token weights and vocabulary mappings are
supported; other layouts fail to load, and recall stays lexical.

## Persisted claim vectors

Claim vectors are saved in `<root>/embeddings/<model>.vec`: a header (format, model
fingerprint, dimension), then one record per claim (SHA-256 of the claim text, float32
unit vector). The fingerprint hashes the tokenizer, the table header, sixteen samples
of the table and an encoding version, so two copies of one model share vectors and a
change to the encoding code invalidates them.

- New vectors are appended. The worker indexes every active claim after processing
  (`"embeddings": {"encoded": n}` in its output), and CLI `recall` saves what it
  encodes, so the hook normally finds every vector on disk.
- The hook encodes at most 512 claims the worker has not indexed per prompt (about
  50 ms), appends them, and stays lexical while a larger backlog remains. A new wiki
  therefore catches up on its own even without the worker.
- The hook never rewrites the file. The worker and CLI rewrite it when it is missing,
  damaged (a torn final record), or stale records (forgotten claims) outnumber live
  ones by more than 1,000, and only they remove other models' files.
- Writes are best effort: a read-only or full disk leaves recall using the vectors it
  computed in memory.

## Hook behaviour

With `JEV_WIKI_EMBEDDING_MODEL` set, the hook loads the model only once recall has
claims, only from local files (a Hugging Face id resolves through the local cache,
following `HF_HUB_CACHE`, `HF_HOME` and `XDG_CACHE_HOME`; nothing is downloaded), and
only while it is within 0.55 s of the hook starting. Loading, encoding, or reading
stored vectors past that point leaves that prompt's recall lexical; the 0.75 s alarm
still ends the hook as before.

Hook wall time, `python -m jev_wiki hook` in a fresh process, one LongMemEval
question's haystack (609 claims):

| | Wall time | Evidence |
|---|---|---|
| Lexical (variable unset) | 0.15–0.17 s | BM25 |
| Embeddings, first prompt with no vectors saved (609 > 512 new) | 0.42 s | BM25 (512 vectors saved) |
| Embeddings, later prompts | 0.27–0.35 s | Hybrid |

In-process, the embedding share was 0.11 s to load and 0.002 s to read the 609 saved
vectors. At 5,906 claims (eight haystacks) it was 0.21 s to load and 0.11 s to read
the vectors, but the hook times out there even with embeddings off: loading and
validating the claims from `state.json` alone takes 0.45 s, so the hook already
returned no evidence at that size. Hook recall on large wikis is bounded by the store
read, not by embeddings.

## Recall quality

The hook calls the same `Engine.recall` with the same embedder, and saved vectors are
the exact float32 values the encoder produced, so hook recall ranks exactly as CLI
recall with the embedder on. [`examples/longmemeval_offline.py`](../examples/longmemeval_offline.py)
with `JEV_WIKI_EMBEDDING_MODEL=default` and the new loader, all 470 questions:

| | @1 | @3 | @5 | @10 | recall_all@10 | Preference @10 |
|---|---|---|---|---|---|---|
| PR #8 (model2vec loader) | 0.866 | 0.951 | 0.966 | 0.983 | 0.936 | 0.967 |
| Memory-mapped loader | 0.866 | 0.951 | 0.966 | 0.983 | 0.936 | 0.967 |

Every type matched PR #8 as well (knowledge-update 1.000, multi-session 1.000,
single-session-assistant 0.946, single-session-user 1.000, temporal-reasoning 0.969).

## Limits

- Each prompt still hashes every active claim and looks up its vector: O(claims),
  about 0.1 s at 6,000 claims. Record keys sit between vectors, so a cold page cache
  reads the whole vector file.
- If the hook and the worker name different models, each keeps its own vector file;
  a worker rewrite removes the hook's.
- The live JEV sweep with the embedder on is still pending on TypeSafe credits.
