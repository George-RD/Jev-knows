"""Okapi BM25 over a small in-memory corpus. Code owns candidate generation."""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    {"the", "a", "an", "and", "or", "to", "of", "is", "are", "was", "what", "how", "i"}
)


def words(text: str) -> list[str]:
    """Casefolded word tokens in order, stopwords removed."""
    return [w for w in re.findall(r"[\w]+", text.casefold()) if w not in STOPWORDS]


def bm25_scores(query: str, docs: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """One BM25 score per document; 0.0 when no query term occurs in it.

    Repeated query terms count once, so a query cannot weight a term by repeating it.
    IDF uses the non-negative Lucene form, so a term present in every document still
    scores slightly above zero instead of cancelling the match.
    """
    tokenized = [words(d) for d in docs]
    n = len(tokenized)
    terms = set(words(query))
    if not n or not terms:
        return [0.0] * n
    avg = sum(map(len, tokenized)) / n or 1.0
    df = Counter(t for doc in tokenized for t in set(doc) & terms)
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
    scores = []
    for doc in tokenized:
        tf = Counter(t for t in doc if t in idf)
        norm = k1 * (1 - b + b * len(doc) / avg)
        scores.append(sum(idf[t] * f * (k1 + 1) / (f + norm) for t, f in tf.items()))
    return scores
