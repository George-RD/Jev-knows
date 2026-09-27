"""BM25 shortlist scoring: pure functions plus the engine's use of them."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jev_wiki.bm25 import bm25_from_stats, bm25_scores, query_terms, term_stats, words
from jev_wiki.engine import Engine
from test_lifecycle import LifecycleDecisionFixture


class Bm25ScoreTests(unittest.TestCase):
    def test_words_casefold_and_drop_stopwords(self):
        self.assertEqual(words("The Café and I ate ÉCLAIRS"), ["café", "ate", "éclairs"])

    def test_no_match_scores_zero(self):
        self.assertEqual(bm25_scores("ornithology", ["blue console", "red kettle"]), [0.0, 0.0])

    def test_empty_inputs(self):
        self.assertEqual(bm25_scores("atlas", []), [])
        self.assertEqual(bm25_scores("the of", ["atlas"]), [0.0])

    def test_rare_term_outranks_several_common_terms(self):
        docs = [f"atlas team notes item {i}" for i in range(20)] + ["priya bakes sourdough"]
        scores = bm25_scores("atlas team notes sourdough", docs)
        self.assertEqual(max(range(len(docs)), key=scores.__getitem__), 20)

    def test_term_in_every_document_still_counts(self):
        scores = bm25_scores("atlas", ["atlas one", "atlas two"])
        self.assertTrue(all(score > 0 for score in scores))

    def test_shorter_document_wins_on_equal_term_frequency(self):
        short, long = bm25_scores("kettle", ["red kettle", "red kettle " + "filler " * 20])
        self.assertGreater(short, long)

    def test_repeated_query_terms_count_once(self):
        docs = ["atlas console", "harbor console"]
        self.assertEqual(bm25_scores("atlas atlas atlas", docs), bm25_scores("atlas", docs))

    def test_term_frequency_saturates(self):
        one, two, many = bm25_scores("atlas", ["atlas x", "atlas atlas", "atlas " * 50])
        self.assertLess(one, two)
        self.assertLess(many, 2.2 * two)

    def test_combined_stats_match_scoring_the_concatenation(self):
        parts = [("red kettle", "Atlas notes"), ("blue console", "Harbor"), ("atlas", "")]
        terms = query_terms("atlas kettle")
        combined = []
        for text, title in parts:
            (a, tf_a), (b, tf_b) = term_stats(text, terms), term_stats(title, terms)
            combined.append((a + b, tf_a + tf_b))
        joined = [f"{text} {title}" for text, title in parts]
        self.assertEqual(bm25_from_stats(combined), bm25_scores("atlas kettle", joined))


class EngineBm25ShortlistTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-bm25-")
        self.addCleanup(temporary.cleanup)
        self.engine = Engine(Path(temporary.name), LifecycleDecisionFixture())

    def test_rare_evidence_survives_a_crowd_of_common_matches(self):
        # 30 claims share four query words; an overlap count ranks all of them above
        # the one claim that holds the rare term, pushing it out of the 24-claim shortlist.
        crowd = [f"Atlas team members think about project item {i}." for i in range(30)]
        text = "\n\n".join(crowd + ["Priya bakes sourdough bread every weekend."])
        self.assertEqual(self.engine.ingest(text, source_key="notes")["status"], "complete")
        result = self.engine.recall(
            "What does the Atlas team think about sourdough?", limit=3, offline=True
        )
        self.assertEqual(result["items"][0]["text"], "Priya bakes sourdough bread every weekend.")

    def test_source_title_takes_part_in_matching(self):
        self.engine.ingest("The console is blue and quiet.", source_key="a", title="Harbor")
        self.engine.ingest("The kettle is red and loud.", source_key="b", title="Atlas")
        items = self.engine.recall("Atlas", offline=True)["items"]
        self.assertEqual([i["text"] for i in items], ["The kettle is red and loud."])


if __name__ == "__main__":
    unittest.main()
