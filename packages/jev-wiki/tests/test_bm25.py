"""BM25 shortlist scoring: pure functions plus the engine's use of them."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jev_wiki.bm25 import bm25_from_stats, bm25_scores, query_terms, term_stats, words
from jev_wiki.engine import SOURCE_CONTEXT_WEIGHT, Engine, _shortlist, _with_source_context
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

    def test_source_on_the_topic_lifts_its_claims(self):
        # Both claims match the same four query words, so the shorter one wins on BM25
        # alone. The other sits in a source that keeps returning to paintings.
        self.engine.ingest(
            "Any ideas for arranging my bookshelf?", source_key="books", title="Books"
        )
        self.engine.ingest(
            "\n\n".join(
                [
                    "Any ideas for what my next canvas should be?",
                    "My paintings are mostly flowers with thick palette knife texture.",
                    "I have started selling my paintings at the Sunday craft market.",
                ]
            ),
            source_key="art",
            title="Art",
        )
        items = self.engine.recall("Any ideas for my paintings?", limit=5, offline=True)["items"]
        texts = [i["text"] for i in items]
        self.assertLess(
            texts.index("Any ideas for what my next canvas should be?"),
            texts.index("Any ideas for arranging my bookshelf?"),
        )

    def test_source_context_never_admits_an_unmatched_claim(self):
        self.engine.ingest(
            "\n\n".join(
                [
                    "The harbor ferry leaves at nine every morning.",
                    "The harbor market sells smoked fish on Fridays.",
                    "Priya repainted the kitchen a pale green colour.",
                ]
            ),
            source_key="town",
        )
        result = self.engine.recall("harbor", limit=5, offline=True)
        self.assertEqual(result["candidate_count"], 2)
        self.assertNotIn(
            "Priya repainted the kitchen a pale green colour.",
            [i["text"] for i in result["items"]],
        )

    def test_strong_plain_match_survives_a_source_repeating_filler(self):
        # One source repeats the filler words 30 times and lifts every claim in it; the
        # only claim holding the rare term must still lead the offline order.
        chat = [f"Any ideas for my kitchen drawer number {i}?" for i in range(30)]
        self.engine.ingest("\n\n".join(chat), source_key="chat")
        groceries = [f"Buy {i} cartons of oat milk at the corner shop." for i in range(20)]
        garden = "Garden plan: tulips along the fence in October."
        self.engine.ingest("\n\n".join([garden, *groceries]), source_key="note")
        result = self.engine.recall("any ideas for my garden", limit=5, offline=True)
        self.assertEqual(result["items"][0]["text"], garden)

    def test_best_plain_matches_always_reach_the_ranker(self):
        # Short filler claims score within 1.5x of the rare-term claim, so the source
        # lift alone would rank all 30 above it and cut it from the 24-claim shortlist.
        chat = [f"Any ideas for my drawer {i}?" for i in range(30)]
        self.engine.ingest("\n\n".join(chat), source_key="chat")
        garden = (
            "My garden needs work: tulips along the long fence in the back yard "
            "this coming October."
        )
        groceries = [f"Buy {i} cartons of oat milk at the corner shop." for i in range(20)]
        self.engine.ingest("\n\n".join([garden, *groceries]), source_key="note")
        provider = self.engine.provider
        provider.calls.clear()
        self.engine.recall("any ideas for my garden", limit=20, max_chars=20_000)
        ranked_state, _ = provider.calls[-1]
        self.assertEqual(len(ranked_state), 24)
        self.assertIn(garden, ranked_state.values())

    def test_shortlist_stays_within_the_byte_budget(self):
        # One sentence each, about 500 bytes: the byte budget binds before 24 claims.
        long_filler = [
            f"Any ideas for my drawer {i}" + " that holds spare cables and old chargers" * 12
            for i in range(30)
        ]
        self.engine.ingest("\n\n".join(long_filler), source_key="chat")
        self.engine.ingest("Garden plan: tulips along the fence.", source_key="note")
        provider = self.engine.provider
        provider.calls.clear()
        self.engine.recall("any ideas for my garden", limit=20, max_chars=20_000)
        ranked_state, _ = provider.calls[-1]
        self.assertIn("Garden plan: tulips along the fence.", ranked_state.values())
        self.assertLessEqual(
            sum(len(t.encode("utf-8")) + 100 for t in ranked_state.values()), 14_000
        )

    def test_weak_match_in_a_strong_source_stays_below_a_strong_match(self):
        self.engine.ingest(
            "\n\n".join(
                [
                    "My paintings are mostly flowers with palette knife texture.",
                    "I sell my paintings at the Sunday craft market.",
                    "My car needs a new set of winter tyres soon.",
                ]
            ),
            source_key="art",
        )
        self.engine.ingest("Paintings from the museum trip were lovely.", source_key="trip")
        diary = [f"My day number {i} was quiet and calm." for i in range(10)]
        self.engine.ingest("\n\n".join(diary), source_key="diary")
        texts = [
            i["text"] for i in self.engine.recall("my paintings", limit=5, offline=True)["items"]
        ]
        self.assertLess(
            texts.index("Paintings from the museum trip were lovely."),
            texts.index("My car needs a new set of winter tyres soon."),
        )

    def test_title_only_match_gets_no_source_lift(self):
        self.engine.ingest(
            "\n\n".join(
                [
                    "Priya repainted the kitchen a pale green colour.",
                    "The spare room still needs new curtains.",
                ]
            ),
            source_key="home",
            title="Paintings",
        )
        self.engine.ingest(
            "My paintings sell best at the Sunday craft market.", source_key="art", title="Art"
        )
        result = self.engine.recall("paintings", limit=5, offline=True)
        self.assertEqual(
            result["items"][0]["text"], "My paintings sell best at the Sunday craft market."
        )

    def test_oversized_candidate_is_skipped_not_the_end_of_the_fill(self):
        def candidate(i, size, plain, lifted):
            text = "x" * size
            return {"id": f"{i:03d}", "text": text, "lexical_score": plain, "context_score": lifted}

        guarded = [candidate(i, 1_000, 10.0 - i / 100, 1.0) for i in range(12)]
        big = candidate(100, 900, 1.0, 5.0)  # Best context score; 13,200 + 1,000 > 14,000.
        small = candidate(101, 50, 0.5, 4.0)
        picked = {c["id"] for c in _shortlist([*guarded, big, small])}
        self.assertNotIn(big["id"], picked)
        self.assertIn(small["id"], picked)
        self.assertEqual(len(picked), 13)

    def test_lift_scales_claim_scores_by_their_source(self):
        terms = query_terms("harbor")
        claims = [{"source_id": "a"}, {"source_id": "a"}, {"source_id": "b"}]
        texts = [term_stats(t, terms) for t in ("harbor ferry", "harbor fish", "quiet lane")]
        titles = {"a": term_stats("", terms), "b": term_stats("Harbor", terms)}
        lifted = _with_source_context(claims, texts, titles, [2.0, 1.0, 0.5])
        best = 1 + SOURCE_CONTEXT_WEIGHT
        self.assertEqual(lifted[:2], [2.0 * best, 1.0 * best])
        self.assertEqual(lifted[2], 0.5)  # Title-only match: no lift.
        self.assertEqual(_with_source_context(claims, texts, titles, [0.0] * 3), [0.0] * 3)


if __name__ == "__main__":
    unittest.main()
