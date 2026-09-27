# Embedding candidates, offline LongMemEval check (2026-09-27)

After the source-context lift
([source-context-shortlist-2026-09-27.md](source-context-shortlist-2026-09-27.md)), six
preference questions still missed at recall_any@10 because the answer shares no useful
words with the question: "What should I serve for dinner this weekend with my
homegrown ingredients?" is answered by "I've been using basil and mint in my cooking
lately" and "I've even harvested some cherry tomatoes from my garden." This change
adds an optional local embedding model as a second candidate source beside BM25. It
was measured without JEV, because the TypeSafe account was out of credits.

## JEV topic labels

Routing by JEV's topic labels could not be tested. Labels come from live intake
calls, and the offline scorer's stub labels every claim `general`. The seven topics
are also coarse ("preferences", "people", ...): "dinner" and "basil" would both land
in `general` or `preferences` with most other claims, so a topic filter narrows little.

## Change

`Engine(root, provider, embedder=...)` takes an optional embedder. The CLI's `recall`
builds one only when `JEV_WIKI_EMBEDDING_MODEL` is set (`default` selects
`minishlab/potion-retrieval-32M`) and the `embed` extra (`model2vec`) is installed;
otherwise recall is unchanged. `recall --offline` never loads a model, since loading
one by name can reach Hugging Face. A model that fails to load, or fails while
encoding, leaves recall lexical rather than failing; a recall that used similarities
reports mode `hybrid`.

With an embedder, recall computes each active claim's cosine similarity to the query:

- The twelve best lexical claims (by source-context score) are taken first, then the
  claims most similar to the query, then further lexical matches if room remains. The
  24-claim and 14 KB bounds are unchanged.
- The shortlist is ordered by reciprocal-rank fusion, `1/(10 + lexical rank) +
  2/(10 + similarity rank)`. JEV reranking breaks relevance ties with the same order.
- A claim below 0.2 similarity is never a semantic candidate, so a query with no
  related memory is not padded with the nearest unrelated claims. Answer-session
  claims had a best similarity of 0.40 / 0.61 / 0.77 (10th / 50th / 90th percentile);
  off-topic probes such as "What is the capital of Peru?" matched no claim at 0.2.

## Choosing the model and mix

Variants were compared with a fast replica of the offline scorer over all 470
questions (the replica's baseline is 0.960 at @10 against the scorer's 0.962; it
skips duplicate-text removal). "Shortlist" is the share of questions with an answer
session anywhere in the 24-claim shortlist, which is what JEV reranking can reach.

| Variant | @1 | @10 | recall_all@10 | Preference @10 | Shortlist | Lost at @10 |
|---|---|---|---|---|---|---|
| BM25 + source lift (`main`) | 0.832 | 0.960 | 0.866 | 0.800 | 0.964 | |
| Embedding only, potion-retrieval-32M | 0.853 | 0.985 | 0.947 | 1.000 | 0.987 | 1 |
| Embedding only, potion-base-8M | 0.866 | 0.974 | 0.940 | 0.900 | 0.987 | 5 |
| Six embedding claims appended to the shortlist | 0.832 | 0.962 | 0.872 | 0.800 | 0.983 | 0 |
| Score blend, context/max + 2 × cosine | 0.885 | 0.974 | 0.921 | 0.867 | 0.977 | 0 |
| RRF over both rankings, one shortlist | 0.877 | 0.970 | 0.911 | 0.833 | 0.974 | 0 |
| 12 lexical + 12 embedding, cosine order | 0.853 | 0.987 | 0.940 | 1.000 | 0.987 | 0 |
| **12 lexical + 12 embedding, RRF (k 10, embedding ×2)** | 0.866 | 0.983 | 0.936 | 0.967 | 0.987 | 0 |
| Same, similarity floor 0.25 | 0.866 | 0.974 | 0.926 | 0.933 | 0.979 | 1 |

On its own the 32M retrieval model beats BM25, but it drops one question BM25 finds.
Reserving half the shortlist for each source keeps every BM25 hit. Cosine order scores
best at @10 but worse at @1; the RRF order was chosen as the balance, and live JEV
reranking reorders the same shortlist anyway. The 8M general model is smaller but
loses five questions on its own.

The one preference question still missed ("tips for my visit to the music store")
has its best answer claim, about open D tuning on an electric guitar, at similarity
rank 14.

## Results

[`examples/longmemeval_offline.py`](../examples/longmemeval_offline.py) with
`JEV_WIKI_EMBEDDING_MODEL=default`, all 470 questions:

| | @1 | @3 | @5 | @10 | recall_all@10 |
|---|---|---|---|---|---|
| `main` (BM25 + source lift) | 0.832 | 0.909 | 0.940 | 0.962 | 0.864 |
| Embedding candidates | 0.866 | 0.951 | 0.966 | 0.983 | 0.936 |

| recall_any@10 by type | `main` | Embedding candidates |
|---|---|---|
| knowledge-update | 1.000 | 1.000 |
| multi-session | 0.975 | 1.000 |
| single-session-assistant | 0.946 | 0.946 |
| single-session-preference | 0.800 | 0.967 |
| single-session-user | 1.000 | 1.000 |
| temporal-reasoning | 0.953 | 0.969 |

No type fell, and the replica found no question lost at @10 (eleven gained). Without
`JEV_WIKI_EMBEDDING_MODEL` the scorer reproduces `main` exactly. The live JEV sweep
with the embedder on is pending on TypeSafe credits.

## Cost

`potion-retrieval-32M` is a static embedding model: a 129 MB download on first use,
cached by Hugging Face, and no torch. Loading takes about 1 s; embedding 555 claims
takes about 30 ms and 10,000 claims about 2.5 s on 4 CPU cores. Vectors are cached per
process only, so each CLI recall pays the load and embeds every active claim. The
prompt hook stays lexical: its 0.75 s budget is shorter than the load. Persisting
vectors beside `state.json` and keeping a loaded model in a long-lived process would
let the hook use embeddings too.
