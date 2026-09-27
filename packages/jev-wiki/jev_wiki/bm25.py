"""Okapi BM25 over a small in-memory corpus. Code owns candidate generation."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

STOPWORDS = frozenset(
    {"the", "a", "an", "and", "or", "to", "of", "is", "are", "was", "what", "how", "i"}
)


def words(text: str) -> list[str]:
    """Casefolded word tokens in order, stopwords removed."""
    return [w for w in re.findall(r"[\w]+", text.casefold()) if w not in STOPWORDS]


def query_terms(query: str) -> frozenset[str]:
    """Distinct query terms: a query cannot weight a term by repeating it."""
    return frozenset(words(query))


def term_stats(text: str, terms: frozenset[str]) -> tuple[int, Counter]:
    """A document's token length and its counts of the given terms only."""
    tokens = words(text)
    return len(tokens), Counter(t for t in tokens if t in terms)


def bm25_from_stats(
    stats: Iterable[tuple[int, Counter]], k1: float = 1.2, b: float = 0.75
) -> list[float]:
    """One BM25 score per ``(length, query-term counts)`` pair; 0.0 when nothing matches.

    Working from per-document statistics keeps memory proportional to the number of
    documents, so a caller can combine parts (claim text plus a shared source title)
    without materialising each concatenation. IDF uses the non-negative Lucene form,
    so a term present in every document still scores slightly above zero.
    """
    stats = list(stats)
    n = len(stats)
    if not n:
        return []
    avg = sum(length for length, _ in stats) / n or 1.0
    df = Counter(t for _, tf in stats for t in tf)
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
    scores = []
    for length, tf in stats:
        norm = k1 * (1 - b + b * length / avg)
        scores.append(sum(idf[t] * f * (k1 + 1) / (f + norm) for t, f in tf.items()))
    return scores


def bm25_scores(query: str, docs: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """One BM25 score per document string; 0.0 when no query term occurs in it."""
    terms = query_terms(query)
    if not terms:
        return [0.0] * len(docs)
    return bm25_from_stats((term_stats(d, terms) for d in docs), k1, b)
