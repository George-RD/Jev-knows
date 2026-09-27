"""Regression checks for independent pair decisions and serving-model provenance."""

import json
import tempfile
import unittest

from jev_wiki.engine import Engine


class ReorderedFixture:
    model = "jev-latest"
    last_model = "jev-1.13.0"

    def __init__(self):
        self.relation_calls = []

    def ask(self, state, questions):
        if next(iter(questions)).startswith("relation"):
            self.relation_calls.append(questions)
            pairs = json.loads(state)["pairs"]
            answers = {}
            for i, key in enumerate(questions):
                pair = pairs[str(i)]
                value = "duplicate" if pair["left"] == pair["right"] else "conflict"
                answers[key] = {"value": value, "confidence": 0.99}
            return dict(reversed(list(answers.items())))
        values = {"keep": "keep", "kind": "fact", "topic": "projects"}
        return {key: {"value": values[key.split("_")[0]], "confidence": 0.99} for key in questions}


class EngineBatchTests(unittest.TestCase):
    def test_reordered_batched_answers_stay_attached_to_the_right_pair(self):
        with tempfile.TemporaryDirectory() as root:
            provider = ReorderedFixture()
            engine = Engine(root, provider)
            for key, colour in (("first", "blue"), ("second", "red"), ("third", "blue")):
                engine.ingest(f"Orion launch colour is {colour}.", key)
            result = engine.maintain()
            self.assertEqual((result["checked_pairs"], result["conflicts"]), (3, 2))
            self.assertEqual(len(provider.relation_calls), 1)
            claims = {c["id"]: c for c in engine.store.claims()}
            for claim in claims.values():
                for relation in claim["relations"]:
                    target = claims[relation["target"]]
                    expected = "duplicate" if claim["text"] == target["text"] else "conflict"
                    self.assertEqual(relation["type"], expected)

    def test_claim_records_actual_serving_model_and_requested_alias(self):
        with tempfile.TemporaryDirectory() as root:
            engine = Engine(root, ReorderedFixture())
            engine.ingest("Orion launch colour is blue.", "orion")
            claim = engine.store.claims()[0]
            self.assertEqual(claim["model"], "jev-1.13.0")
            self.assertEqual(claim["requested_model"], "jev-latest")


if __name__ == "__main__":
    unittest.main()
