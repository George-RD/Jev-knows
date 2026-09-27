"""Optional local embeddings: a second recall candidate source beside BM25.

Lexical matching cannot connect "dinner with my homegrown ingredients" to evidence
that only says "basil" and "cherry tomatoes". A small static embedding model can, on
CPU and without a network call per query. It is off unless the caller passes an
embedder, and ``from_env`` builds one only when ``JEV_WIKI_EMBEDDING_MODEL`` is set
and the ``embed`` extra (model2vec) is installed.
"""

from __future__ import annotations

import os
from typing import Any

ENV_VAR = "JEV_WIKI_EMBEDDING_MODEL"
# Chosen on LongMemEval_S; see docs/embedding-candidates-2026-09-27.md.
DEFAULT_MODEL = "minishlab/potion-retrieval-32M"


class StaticEmbedder:
    """Cosine similarity through a model2vec static embedding model.

    Loading a model name downloads it from Hugging Face once and caches it; a local
    directory path loads without the network. Vectors are cached per text for the
    life of the instance.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        import numpy as np
        from model2vec import StaticModel

        self.name = model
        self._np = np
        self._model = StaticModel.from_pretrained(model)
        self._cache: dict[str, Any] = {}

    def _unit(self, texts: list[str]) -> Any:
        np = self._np
        vectors = np.asarray(self._model.encode(texts), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def similarities(self, query: str, texts: list[str]) -> list[float]:
        """Cosine similarity of the query to each text, in input order."""
        if not texts:
            return []
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            self._cache.update(zip(missing, self._unit(missing)))
        matrix = self._np.stack([self._cache[t] for t in texts])
        return (matrix @ self._unit([query])[0]).tolist()


def from_env() -> StaticEmbedder | None:
    """The embedder named by JEV_WIKI_EMBEDDING_MODEL, or None when unset or unavailable.

    ``default`` selects DEFAULT_MODEL. A missing extra or a model that fails to load
    returns None so recall stays lexical rather than failing.
    """
    name = os.environ.get(ENV_VAR, "").strip()
    if not name:
        return None
    try:
        return StaticEmbedder(DEFAULT_MODEL if name == "default" else name)
    except Exception:  # noqa: BLE001 - optional accelerator; lexical recall still works.
        return None
