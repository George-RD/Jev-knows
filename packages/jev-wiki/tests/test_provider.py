"""Wire-contract and failure-boundary tests; no JEV credentials or network needed."""

import copy
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from jev_wiki.provider import JevProvider, ProviderError, ScriptedProvider, TransportResponse


def choice(prompt="Is the claim supported?"):
    return {
        "type": "choice",
        "prompt": prompt,
        "choices": {"supported": "Direct source support", "unknown": "Not stated"},
    }


def response_for(payload):
    answers = {}
    for name, question in payload["questions"].items():
        kind = question["type"]
        if kind == "choice":
            labels = list(question["criteria"])
            answers[name] = {
                "type": kind,
                "choice": labels[0],
                "confidence": 0.8,
                "probabilities": {label: float(i == 0) for i, label in enumerate(labels)},
            }
        elif kind == "score":
            answers[name] = {
                "type": kind,
                "score": 0.75,
                "confidence": 0.5,
                "legend": {str(i): text for i, text in enumerate(question["criteria"])},
                "probabilities": {"0": 0.25, "1": 0.75},
            }
        else:
            answers[name] = {"type": kind, "noul": 0.73}
    return {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


class RecordingTransport:
    def __init__(self, actions=None):
        self.requests = []
        self.actions = list(actions or [])

    def __call__(self, request, timeout):
        payload = json.loads(request.data)
        self.requests.append((request, timeout, payload))
        action = self.actions.pop(0) if self.actions else None
        if isinstance(action, Exception):
            raise action
        if isinstance(action, TransportResponse):
            return action
        body = response_for(payload)
        if callable(action):
            action(body)
        return TransportResponse(200, json.dumps(body).encode(), {})


class ProviderTests(unittest.TestCase):
    def test_real_native_wire_contract_all_three_types(self):
        transport = RecordingTransport()
        provider = JevProvider("test-key", transport=transport)
        questions = {
            "claim": choice(),
            "importance": {"type": "score", "prompt": "How useful?", "levels": ["Low", "High"]},
            "personal": {
                "type": "noul",
                "prompt": "Is this personal information?",
                "criteria": {"true": "Personal", "false": "Public"},
            },
        }
        answers = provider.ask("Verbatim source", questions)
        request, timeout, body = transport.requests[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(timeout, 15)
        self.assertEqual(body["state"], "Verbatim source")
        self.assertEqual(body["model"], "jev-1.13.0")
        self.assertEqual(body["questions"]["claim"]["instructions"], questions["claim"]["prompt"])
        self.assertEqual(body["questions"]["claim"]["criteria"], questions["claim"]["choices"])
        self.assertNotIn("prompt", body["questions"]["claim"])
        self.assertEqual(answers["claim"]["value"], "supported")
        self.assertEqual(answers["importance"]["value"], 0.75)
        self.assertEqual(answers["personal"], {"value": 0.73, "confidence": None})
        self.assertEqual(provider.telemetry["input_tokens"], 100)
        self.assertEqual(provider.last_model, "jev-1.13.0")

    def test_credentials_are_required_no_silent_fallback(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ProviderError):
            JevProvider()
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "environment-key"}):
            transport = RecordingTransport()
            JevProvider(transport=transport).ask("source", {"q": choice()})
            self.assertEqual(
                transport.requests[0][0].get_header("Authorization"), "Bearer environment-key"
            )

    def test_invalid_questions_never_reach_transport(self):
        invalid = [
            {"type": "text", "prompt": "Generate something"},
            {"type": "choice", "prompt": "", "choices": {"a": "A", "b": "B"}},
            {"type": "choice", "prompt": "Pick", "choices": {str(i): "" for i in range(256)}},
            {"type": "score", "prompt": "Rate", "levels": ["Only one"]},
            {"type": "noul", "prompt": "Yes?", "criteria": {"yes": "yes"}},
            {"type": "noul", "prompt": "Yes?", "unexpected": "not in schema"},
        ]
        transport = RecordingTransport()
        provider = JevProvider("key", transport=transport)
        for question in invalid:
            with self.subTest(question=question), self.assertRaises(ProviderError):
                provider.ask("source", {"q": question})
        self.assertEqual(transport.requests, [])

    def test_bounded_questions_share_state_and_all_answers_are_returned(self):
        transport = RecordingTransport()
        provider = JevProvider("key", transport=transport)
        questions = {f"q{i}": choice() for i in range(260)}
        answers = provider.ask("One shared source", questions)
        self.assertEqual(set(answers), set(questions))
        self.assertEqual([len(item[2]["questions"]) for item in transport.requests], [128, 128, 4])
        self.assertTrue(all(item[2]["state"] == "One shared source" for item in transport.requests))
        self.assertEqual(provider.telemetry["requests"], 3)

    def test_byte_budget_splits_before_question_count_limit(self):
        transport = RecordingTransport()
        provider = JevProvider("key", transport=transport)
        questions = {f"q{i}": choice("Describe this? " + "x" * 1000) for i in range(100)}
        provider.ask("source", questions)
        self.assertGreater(len(transport.requests), 1)
        self.assertTrue(
            all(len(item[0].data) <= provider.MAX_REQUEST_BYTES for item in transport.requests)
        )

    def test_late_oversized_question_rejects_all_before_io(self):
        transport = RecordingTransport()
        provider = JevProvider("key", transport=transport)
        questions = {f"q{i}": choice() for i in range(130)}
        questions["too_large"] = choice("x" * 29_000)
        with self.assertRaises(ProviderError):
            provider.ask("source", questions)
        self.assertEqual(transport.requests, [])

    def test_invalid_answers_fail_closed_without_retry(self):
        mutations = [
            lambda body: body["answers"].clear(),
            lambda body: body["answers"].update(extra=body["answers"]["q"]),
            lambda body: body.update(model="jev-1.14.0"),
            lambda body: body["answers"]["q"].update(type="noul"),
            lambda body: body["answers"]["q"].update(choice="invented"),
            lambda body: body["answers"]["q"].update(confidence=True),
            lambda body: body["answers"]["q"].update(confidence=float("nan")),
            lambda body: body["answers"]["q"].update(
                probabilities={"supported": 0.1, "unknown": 0.1}
            ),
            lambda body: body["usage"].update(input_tokens=-1),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                transport = RecordingTransport([mutate])
                provider = JevProvider("key", transport=transport)
                with self.assertRaises(ProviderError):
                    provider.ask("source", {"q": choice()})
                self.assertEqual(len(transport.requests), 1)
                self.assertEqual(provider.telemetry["failures"], 1)

    def test_two_decimal_rounding_drift_is_accepted(self):
        # Observed live: seven options rounded to hundredths summing to 0.99.
        labels = {f"l{i}": f"Label {i}" for i in range(7)}
        question = {"type": "choice", "prompt": "Pick one.", "choices": labels}
        rounded = dict(zip(labels, (0.81, 0.16, 0.01, 0.01, 0.0, 0.0, 0.0)))

        def mutate(body):
            body["answers"]["q"].update(choice="l0", probabilities=rounded)

        provider = JevProvider("key", transport=RecordingTransport([mutate]))
        self.assertEqual(provider.ask("source", {"q": question})["q"]["value"], "l0")

    def test_noul_probability_not_boolean_and_invalid_score_rejected(self):
        mutations_and_questions = [
            (
                lambda body: body["answers"]["q"].update(noul=True),
                {"type": "noul", "prompt": "Yes?"},
            ),
            (
                lambda body: body["answers"]["q"].update(score=10),
                {"type": "score", "prompt": "Rate", "levels": ["Low", "High"]},
            ),
        ]
        for mutate, question in mutations_and_questions:
            with self.subTest(question=question), self.assertRaises(ProviderError):
                JevProvider("key", transport=RecordingTransport([mutate])).ask(
                    "source", {"q": question}
                )

    def test_transient_errors_retry_with_bounded_retry_after(self):
        transport = RecordingTransport(
            [
                TransportResponse(429, b"sensitive server body", {"Retry-After": "99999"}),
                TransportResponse(529, b"overloaded", {}),
            ]
        )
        delays = []
        provider = JevProvider("key", transport=transport, sleep=delays.append)
        provider.ask("source", {"q": choice()})
        self.assertEqual(delays, [5.0, 1.0])
        self.assertEqual(provider.telemetry["retries"], 2)
        self.assertEqual(provider.telemetry["requests"], 3)

    def test_permanent_errors_do_not_retry_or_expose_bodies(self):
        for status in [301, 401, 403, 404, 422]:
            transport = RecordingTransport(
                [TransportResponse(status, b"secret-key source-text", {})]
            )
            provider = JevProvider("secret-key", transport=transport)
            with self.subTest(status=status), self.assertRaises(ProviderError) as caught:
                provider.ask("source-text", {"q": choice()})
            self.assertNotIn("secret-key", str(caught.exception))
            self.assertNotIn("source-text", str(caught.exception))
            self.assertEqual(len(transport.requests), 1)

    def test_network_failures_exhaust_three_attempts_and_redact_errors(self):
        transport = RecordingTransport([urllib.error.URLError("secret-key") for _ in range(3)])
        provider = JevProvider("secret-key", transport=transport, sleep=lambda _: None)
        with self.assertRaises(ProviderError) as caught:
            provider.ask("source", {"q": choice()})
        self.assertNotIn("secret-key", str(caught.exception))
        self.assertEqual(len(transport.requests), 3)

    def test_invalid_json_is_not_retried(self):
        transport = RecordingTransport([TransportResponse(200, b"not-json", {})])
        with self.assertRaises(ProviderError):
            JevProvider("key", transport=transport).ask("source", {"q": choice()})
        self.assertEqual(len(transport.requests), 1)

    def test_second_batch_failure_does_not_return_partial_decisions(self):
        transport = RecordingTransport([None, TransportResponse(422, b"invalid", {})])
        provider = JevProvider("key", transport=transport)
        with self.assertRaises(ProviderError):
            provider.ask("source", {f"q{i}": choice() for i in range(129)})
        self.assertEqual(len(transport.requests), 2)

    def test_versioned_cache_is_private_and_prompt_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = RecordingTransport()
            provider = JevProvider("secret-key", transport=transport, cache_dir=directory)
            first = provider.ask("source-private", {"q": choice()})
            second = provider.ask("source-private", {"q": choice()})
            self.assertEqual(first, second)
            self.assertEqual(len(transport.requests), 1)
            self.assertEqual(provider.telemetry["cache_hits"], 1)
            self.assertEqual(provider.telemetry["input_tokens"], 100)
            cache_files = list(Path(directory).glob("*.json"))
            self.assertEqual(len(cache_files), 1)
            text = cache_files[0].read_text()
            self.assertNotIn("secret-key", text)
            self.assertNotIn("source-private", text)
            self.assertEqual(cache_files[0].stat().st_mode & 0o777, 0o600)
            provider.ask("different source", {"q": choice()})
            provider.ask("source-private", {"q": choice("A changed rubric?")})
            self.assertEqual(len(transport.requests), 3)

    def test_corrupt_cache_is_validated_then_refetched(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = RecordingTransport()
            provider = JevProvider("key", transport=transport, cache_dir=directory)
            provider.ask("source", {"q": choice()})
            cache = next(Path(directory).glob("*.json"))
            cache.write_text('{"answers":{"q":{"value":"fabricated"}}}')
            result = provider.ask("source", {"q": choice()})
            self.assertEqual(result["q"]["value"], "supported")
            self.assertEqual(provider.telemetry["cache_errors"], 1)
            self.assertEqual(len(transport.requests), 2)

    def test_model_alias_does_not_use_stale_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = RecordingTransport()
            provider = JevProvider(
                "key", model="jev-latest", transport=transport, cache_dir=directory
            )
            provider.ask("source", {"q": choice()})
            provider.ask("source", {"q": choice()})
            self.assertEqual(len(transport.requests), 2)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_empty_questions_make_no_request(self):
        transport = RecordingTransport()
        self.assertEqual(JevProvider("key", transport=transport).ask("source", {}), {})
        self.assertEqual(transport.requests, [])

    def test_explicit_scripted_provider_exhaustion_is_an_error(self):
        answers = {"q": {"value": "supported", "confidence": 1.0}}
        provider = ScriptedProvider([copy.deepcopy(answers)])
        self.assertEqual(provider.ask("source", {"q": choice()}), answers)
        with self.assertRaises(ProviderError):
            provider.ask("source", {"q": choice()})


if __name__ == "__main__":
    unittest.main()
