"""Sentence-level claim candidates keep exact offsets and avoid fragment claims."""

import unittest

from jev_wiki.engine import MIN_CLAIM_CHARS, candidate_spans


class CandidateSpanTests(unittest.TestCase):
    def texts(self, text, **kwargs):
        spans = candidate_spans(text, **kwargs)
        for span in spans:
            self.assertEqual(text[span["start"] : span["end"]], span["text"])
        return [span["text"] for span in spans]

    def test_chatty_paragraph_splits_personal_fact_from_request(self):
        text = (
            "Hey, quick question about dinner tonight. I've been vegetarian since 2019 and "
            "my partner is allergic to peanuts. Can you suggest three easy recipes?"
        )
        self.assertEqual(
            self.texts(text),
            [
                "Hey, quick question about dinner tonight.",
                "I've been vegetarian since 2019 and my partner is allergic to peanuts.",
                "Can you suggest three easy recipes?",
            ],
        )

    def test_known_abbreviations_do_not_end_a_sentence(self):
        text = "Dr. Patel moved our check-up, e.g. to March, and Mr. Smith agreed to it."
        self.assertEqual(self.texts(text), [text])

    def test_numbers_and_single_letters_still_end_a_sentence(self):
        text = "My daughter just turned 7. Can you suggest a party theme for vitamin D."
        self.assertEqual(
            self.texts(text),
            ["My daughter just turned 7.", "Can you suggest a party theme for vitamin D."],
        )

    def test_short_fragments_join_a_neighbour(self):
        text = "Thanks! I adopted a greyhound named Pip last week.\nOk."
        texts = self.texts(text)
        self.assertEqual(texts, ["Thanks! I adopted a greyhound named Pip last week.\nOk."])
        self.assertTrue(all(len(t) >= MIN_CLAIM_CHARS for t in texts))

    def test_line_breaks_and_paragraphs_are_boundaries(self):
        text = "- I drive a blue Corolla to work\n- My office moved to Leeds in May\n\nIt rained."
        self.assertEqual(
            self.texts(text),
            ["- I drive a blue Corolla to work", "- My office moved to Leeds in May", "It rained."],
        )

    def test_long_sentences_are_bounded_at_whitespace(self):
        text = "I keep " + "many old vinyl records " * 60 + "at home."
        texts = self.texts(text, max_chars=200)
        self.assertGreater(len(texts), 1)
        self.assertTrue(all(len(t) <= 200 for t in texts))
        self.assertEqual(" ".join(texts), text)

    def test_quoted_sentence_end_keeps_closing_quote(self):
        text = 'My sister always says "measure twice." She is a carpenter in Bristol.'
        self.assertEqual(
            self.texts(text),
            ['My sister always says "measure twice."', "She is a carpenter in Bristol."],
        )


if __name__ == "__main__":
    unittest.main()
