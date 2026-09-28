"""Optional embedding candidates: recall beyond shared words, without losing lexical matches."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jev_wiki import embedding
from jev_wiki.engine import (
    MIN_SIMILARITY,
    PLAIN_BM25_GUARD,
    SHORTLIST_BYTES,
    SHORTLIST_SIZE,
    Engine,
    _candidates,
    _shortlist,
)
from test_lifecycle import LifecycleDecisionFixture

GARDEN = {"homegrown", "dinner", "basil", "tomatoes", "garden", "herbs"}


class ConceptEmbedder:
    """Similarity is the share of a text's words in the query's concept, else a floor."""

    def __init__(self, concept=GARDEN, background=0.05):
        self.concept, self.background = concept, background
        self.calls = 0

    def similarities(self, query, texts):
        self.calls += 1
        if not set(query.casefold().replace("?", "").split()) & self.concept:
            return [self.background] * len(texts)
        out = []
        for text in texts:
            words = set(text.casefold().replace(".", "").split())
            out.append(max(self.background, len(words & self.concept) / 2))
        return out


class EmbeddingRecallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-embed-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.embedder = ConceptEmbedder()
        self.engine = Engine(self.root, LifecycleDecisionFixture(), embedder=self.embedder)
        text = "\n\n".join(
            [
                "I grow basil and cherry tomatoes on the balcony.",
                "Ideas for a quick weeknight meal would help.",
                "The quarterly report is due next Friday.",
            ]
        )
        self.assertEqual(self.engine.ingest(text, source_key="notes")["status"], "complete")

    def test_similar_claim_without_shared_words_is_recalled(self):
        query = "Suggest a dinner with my homegrown ingredients"
        texts = [i["text"] for i in self.engine.recall(query, offline=True)["items"]]
        self.assertIn("I grow basil and cherry tomatoes on the balcony.", texts)
        lexical_only = Engine(self.root).recall(query, offline=True)["items"]
        self.assertNotIn(
            "I grow basil and cherry tomatoes on the balcony.", [i["text"] for i in lexical_only]
        )

    def test_unrelated_query_is_not_padded_with_nearest_claims(self):
        result = self.engine.recall("Which ferry leaves at nine?", offline=True)
        self.assertEqual(self.embedder.calls, 1)
        self.assertEqual(result["items"], [])
        self.assertEqual(result["candidate_count"], 0)

    def test_embedder_failure_leaves_recall_lexical(self):
        class Broken:
            def similarities(self, query, texts):
                raise RuntimeError("encoder failed")

        engine = Engine(self.root, embedder=Broken())
        result = engine.recall("quarterly report", offline=True)
        self.assertEqual(result["mode"], "lexical")
        self.assertEqual(
            [i["text"] for i in result["items"]], ["The quarterly report is due next Friday."]
        )

    def test_embedding_recall_reports_hybrid_mode(self):
        self.assertEqual(self.engine.recall("basil", offline=True)["mode"], "hybrid")

    def test_similar_claim_ranks_above_a_weak_lexical_match(self):
        items = self.engine.recall("Any basil dinner ideas?", offline=True)["items"]
        self.assertEqual(items[0]["text"], "I grow basil and cherry tomatoes on the balcony.")


class EmbeddingShortlistTests(unittest.TestCase):
    @staticmethod
    def claims(n, prefix, size=40):
        return [{"id": f"{prefix}{i:03d}", "source_id": "s", "text": "x" * size} for i in range(n)]

    def test_best_lexical_matches_keep_their_slots(self):
        lexical = self.claims(20, "l")
        similar = self.claims(30, "e")
        claims = [*lexical, *similar]
        scores = [20.0 - i for i in range(20)] + [0.0] * 30
        similarities = [0.0] * 20 + [0.9 - i / 100 for i in range(30)]
        picked = _shortlist(_candidates(claims, scores, scores, similarities))
        ids = {c["id"] for c in picked}
        self.assertEqual(len(picked), SHORTLIST_SIZE)
        self.assertTrue({c["id"] for c in lexical[:PLAIN_BM25_GUARD]} <= ids)
        self.assertTrue({c["id"] for c in similar[: SHORTLIST_SIZE - PLAIN_BM25_GUARD]} <= ids)

    def test_rare_plain_match_survives_lifted_filler_and_similar_claims(self):
        filler = self.claims(12, "f")
        rare = {"id": "rare", "source_id": "s", "text": "x" * 40}
        similar = self.claims(12, "e")
        claims = [*filler, rare, *similar]
        plain = [1.0] * 12 + [5.0] + [0.0] * 12
        lifted = [9.0] * 12 + [5.0] + [0.0] * 12
        similarities = [0.0] * 13 + [0.9] * 12
        picked = {c["id"] for c in _shortlist(_candidates(claims, plain, lifted, similarities))}
        self.assertIn("rare", picked)

    def test_lexical_matches_fill_room_the_similar_claims_leave(self):
        claims = self.claims(30, "l")
        scores = [30.0 - i for i in range(30)]
        picked = _shortlist(_candidates(claims, scores, scores, [0.0] * 30))
        self.assertEqual(len(picked), SHORTLIST_SIZE)
        self.assertTrue(all(c["semantic_score"] < MIN_SIMILARITY for c in picked))

    def test_mixed_shortlist_stays_within_the_byte_budget(self):
        claims = self.claims(40, "c", size=1_000)
        scores = [float(i % 2) for i in range(40)]
        similarities = [0.5] * 40
        picked = _shortlist(_candidates(claims, scores, scores, similarities))
        cost = sum(len(c["text"].encode()) + 100 for c in picked)
        self.assertLessEqual(cost, SHORTLIST_BYTES)
        self.assertEqual(len(picked), SHORTLIST_BYTES // 1_100)

    def test_small_similar_claims_backfill_past_oversized_ones(self):
        big = self.claims(SHORTLIST_SIZE, "b", size=3_000)
        small = self.claims(10, "s")
        claims = [*big, *small]
        similarities = [0.9] * len(big) + [0.5] * len(small)
        picked = {
            c["id"] for c in _shortlist(_candidates(claims, [0.0] * 34, [0.0] * 34, similarities))
        }
        self.assertTrue({c["id"] for c in small} <= picked)

    def test_without_similarities_candidates_are_the_lexical_matches(self):
        claims = self.claims(3, "c")
        candidates = _candidates(claims, [1.0, 0.0, 2.0], [1.0, 0.0, 2.0])
        self.assertEqual([c["id"] for c in candidates], ["c000", "c002"])
        self.assertTrue(all("semantic_score" not in c for c in candidates))


class CliEmbedderTests(unittest.TestCase):
    def recall(self, *flags):
        from jev_wiki import cli

        with tempfile.TemporaryDirectory() as root:
            self.root = Path(root)
            args = cli._parser().parse_args(
                ["--root", root, "--provider", "none", "recall", "basil", *flags]
            )
            with mock.patch.object(embedding, "from_env", return_value=None) as built:
                cli.run(args)
        return built

    def test_recall_builds_the_configured_embedder(self):
        self.recall().assert_called_once_with(cache_dir=self.root / embedding.VECTOR_DIR)

    def test_offline_recall_never_loads_a_model(self):
        self.recall("--offline").assert_not_called()


class FromEnvTests(unittest.TestCase):
    def test_unset_means_no_embedder(self):
        with mock.patch.dict(os.environ, {embedding.ENV_VAR: ""}):
            self.assertIsNone(embedding.from_env())

    def test_unloadable_model_falls_back_to_lexical(self):
        failing = mock.Mock(side_effect=ImportError("model2vec"))
        with (
            mock.patch.dict(os.environ, {embedding.ENV_VAR: "default"}),
            mock.patch.object(embedding, "StaticEmbedder", failing),
        ):
            self.assertIsNone(embedding.from_env())
        failing.assert_called_once_with(
            embedding.DEFAULT_MODEL, local_only=False, cache_dir=None, deadline=None, max_new=None
        )


if __name__ == "__main__":
    unittest.main()
