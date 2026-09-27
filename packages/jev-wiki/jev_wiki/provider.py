"""Bounded, validated access to JEV's decision API; no generative fallback.

Wire contract verified against https://docs.typesafe.ai/api and /models.
Byte budgets are deliberately conservative: they are not token estimates.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import stat
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ProviderError(RuntimeError):
    """The decision service could not produce a complete validated answer set."""


class Provider(Protocol):
    def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]: ...


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


Transport = Callable[[urllib.request.Request, float], TransportResponse]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


MAX_RESPONSE_BYTES = 4_000_000


def _http_transport(request: urllib.request.Request, timeout: float) -> TransportResponse:
    # Never forward an API credential to a redirect target.
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ProviderError("JEV response exceeded the response-size limit")
        return TransportResponse(response.code, body, dict(response.headers.items()))


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ProviderError("Decision request contains invalid JSON data") from error


def _number(value: object, low: float, high: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderError(f"JEV response has a nonnumeric {field}")
    # Compare before float conversion so enormous JSON integers cannot overflow.
    if not low <= value <= high:
        raise ProviderError(f"JEV response has an out-of-range {field}")
    return float(value)


def _wire_questions(questions: dict[str, dict]) -> dict[str, dict]:
    if not isinstance(questions, dict):
        raise ProviderError("Questions must be a dictionary")
    if len(questions) > 4096:
        raise ProviderError("A single ask is limited to 4096 questions")
    result = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name or len(name) > 200:
            raise ProviderError("Question identifiers must be 1–200 characters")
        if not isinstance(question, dict):
            raise ProviderError("Each question must be a dictionary")
        kind = question.get("type")
        prompt = question.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ProviderError("Each question needs an explicit nonempty prompt")
        wire = {"type": kind, "instructions": prompt}
        if kind == "choice":
            allowed = {"type", "prompt", "choices"}
            choices = question.get("choices")
            if not isinstance(choices, dict) or not 2 <= len(choices) <= 255:
                raise ProviderError("Choice questions require 2–255 choices")
            if any(
                not isinstance(key, str) or not key or not isinstance(description, str)
                for key, description in choices.items()
            ):
                raise ProviderError("Choice labels and descriptions must be strings")
            wire["criteria"] = dict(choices)
        elif kind == "score":
            allowed = {"type", "prompt", "levels"}
            levels = question.get("levels")
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                raise ProviderError("Score questions require 2–10 levels")
            if any(not isinstance(level, str) or not level.strip() for level in levels):
                raise ProviderError("Score levels need nonempty descriptions")
            wire["criteria"] = list(levels)
        elif kind == "noul":
            allowed = {"type", "prompt", "criteria"}
            criteria = question.get("criteria")
            if criteria is not None:
                if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
                    raise ProviderError("Noul criteria may contain only true and false")
                if any(not isinstance(value, str) for value in criteria.values()):
                    raise ProviderError("Noul criteria descriptions must be strings")
                wire["criteria"] = dict(criteria)
        else:
            raise ProviderError("Unknown question type; use choice, score, or noul")
        if set(question) - allowed:
            raise ProviderError("Question contains unsupported fields")
        result[name] = wire
    return result


def _validated_answers(raw: object, questions: dict[str, dict], model: str) -> dict[str, dict]:
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), str):
        raise ProviderError("JEV response is missing its model")
    if model not in {"jev-latest", "jev-preview"} and raw["model"] != model:
        raise ProviderError("JEV response model does not match the pinned model")
    if not re.fullmatch(r"jev-\d+\.\d+\.\d+", raw["model"]):
        raise ProviderError("JEV response must identify a versioned model")
    answers = raw.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ProviderError("JEV response question identifiers do not match the request")
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise ProviderError("JEV response is missing usage")
    for field in ("input_tokens", "output_tokens"):
        if type(usage.get(field)) is not int or usage[field] < 0:
            raise ProviderError("JEV response contains invalid token usage")
    result = {}
    for name, question in questions.items():
        answer = answers[name]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ProviderError("JEV response answer type does not match its question")
        if kind == "noul":
            result[name] = {
                "value": _number(answer.get("noul"), 0, 1, "noul"),
                "confidence": None,
            }
            continue
        confidence = _number(answer.get("confidence"), 0, 1, "confidence")
        expected = (
            set(question["criteria"])
            if kind == "choice"
            else {str(i) for i in range(len(question["criteria"]))}
        )
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != expected:
            raise ProviderError("JEV response probabilities do not match the criteria")
        probabilities = {
            key: _number(value, 0, 1, "probability") for key, value in probabilities.items()
        }
        # The service rounds each probability to two decimals, so the sum can
        # drift by up to half a hundredth per option.
        tolerance = 0.005 * len(probabilities) + 1e-9
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=tolerance):
            raise ProviderError("JEV response probabilities do not sum to one")
        if kind == "choice":
            value = answer.get("choice")
            if not isinstance(value, str) or value not in expected:
                raise ProviderError("JEV response selected an unknown choice")
            if probabilities[value] + 0.005 < max(probabilities.values()):
                raise ProviderError("JEV response choice is not the highest-probability option")
        else:
            value = _number(answer.get("score"), 0, len(expected) - 1, "score")
            legend = answer.get("legend")
            if legend != {str(i): level for i, level in enumerate(question["criteria"])}:
                raise ProviderError("JEV response score legend does not match the criteria")
            expectation = sum(int(key) * probability for key, probability in probabilities.items())
            if not math.isclose(value, expectation, abs_tol=0.025):
                raise ProviderError(
                    "JEV response score does not match its probability distribution"
                )
        result[name] = {
            "value": value,
            "confidence": confidence,
            "probabilities": probabilities,
        }
    return result


def _cache_record(raw: dict, questions: dict[str, dict]) -> dict:
    """Persist only fields already checked by _validated_answers.

    A valid envelope can still include untrusted debug echoes of source text or
    credentials. Those extra fields must not turn the response cache into a log.
    """
    fields = {
        "choice": ("type", "choice", "confidence", "probabilities"),
        "score": ("type", "score", "confidence", "probabilities", "legend"),
        "noul": ("type", "noul"),
    }
    return {
        "model": raw["model"],
        "answers": {
            name: {key: raw["answers"][name][key] for key in fields[question["type"]]}
            for name, question in questions.items()
        },
        "usage": {key: raw["usage"][key] for key in ("input_tokens", "output_tokens")},
    }


class JevProvider:
    """Real JEV provider. Calls are synchronous; independently named questions batch.

    ``transport`` and ``sleep`` are injection seams for explicit test doubles.
    Credentials, source state and prompts are never written to the response cache.
    Aliases deliberately bypass caching because their target version can change.
    """

    ENDPOINT = "https://api.typesafe.ai/v1/systemone"
    CACHE_VERSION = 1
    MAX_REQUEST_BYTES = 56_000
    MAX_STATE_QUESTION_BYTES = 28_000
    MAX_QUESTIONS_PER_REQUEST = 128
    MAX_ATTEMPTS = 3
    RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "jev-1.13.0",
        timeout: float = 15,
        cache_dir: str | Path | None = None,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = os.environ.get("TYPESAFE_API_KEY") if api_key is None else api_key
        if not isinstance(key, str) or not key.strip() or any(c in key for c in "\r\n"):
            raise ProviderError("Set TYPESAFE_API_KEY to use the JEV provider")
        if not isinstance(model, str) or not re.fullmatch(
            r"jev-(?:\d+\.\d+\.\d+|latest|preview)", model
        ):
            raise ProviderError("Use a versioned JEV model or jev-latest/jev-preview")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ProviderError("Timeout must be a positive number")
        if not 0 < timeout <= 120 or not math.isfinite(timeout):
            raise ProviderError("Timeout must be greater than zero and at most 120 seconds")
        self._api_key = key
        self.model = model
        self.timeout = float(timeout)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._transport = transport or _http_transport
        self._sleep = sleep
        self.telemetry = {
            "asks": 0,
            "questions": 0,
            "requests": 0,
            "retries": 0,
            "cache_hits": 0,
            "cache_errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "failures": 0,
        }
        self.last_model: str | None = None

    def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        self.telemetry["asks"] += 1
        try:
            if not isinstance(state, str):
                raise ProviderError("State must be a string")
            wire = _wire_questions(questions)
            self.telemetry["questions"] += len(wire)
            # Validate and plan every batch before making any network request.
            batches = self._batches(state, wire)
            result = {}
            for batch in batches:
                result.update(self._ask_batch(state, batch))
            return result
        except ProviderError:
            self.telemetry["failures"] += 1
            raise

    def _batches(self, state: str, questions: dict[str, dict]) -> list[dict[str, dict]]:
        batches: list[dict[str, dict]] = []
        current: dict[str, dict] = {}
        for name, question in questions.items():
            one = {"model": self.model, "state": state, "questions": {name: question}}
            if len(_json_bytes(one)) > self.MAX_STATE_QUESTION_BYTES:
                raise ProviderError(
                    "State plus question is too large; split source text before asking JEV"
                )
            candidate = {**current, name: question}
            payload = {"model": self.model, "state": state, "questions": candidate}
            if current and (
                len(candidate) > self.MAX_QUESTIONS_PER_REQUEST
                or len(_json_bytes(payload)) > self.MAX_REQUEST_BYTES
            ):
                batches.append(current)
                current = {name: question}
            else:
                current = candidate
        if current:
            batches.append(current)
        return batches

    def _ask_batch(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        payload = {"model": self.model, "state": state, "questions": questions}
        encoded = _json_bytes(payload)
        cache_path = self._cache_path(encoded)
        if cache_path is not None:
            try:
                if cache_path.exists():
                    raw = self._read_cache(cache_path)
                    result = _validated_answers(raw, questions, self.model)
                    clean = _cache_record(raw, questions)
                    if clean != raw:
                        self._write_cache(cache_path, clean)
                    self.telemetry["cache_hits"] += 1
                    self.last_model = raw["model"]
                    return result
            except (OSError, ValueError, ProviderError):
                # A corrupt optional cache is a miss, never an unchecked decision.
                self.telemetry["cache_errors"] += 1
        raw = self._request(encoded)
        result = _validated_answers(raw, questions, self.model)
        self.telemetry["input_tokens"] += raw["usage"]["input_tokens"]
        self.telemetry["output_tokens"] += raw["usage"]["output_tokens"]
        self.last_model = raw["model"]
        if cache_path is not None:
            self._write_cache(cache_path, _cache_record(raw, questions))
        return result

    def _cache_path(self, payload: bytes) -> Path | None:
        if self.cache_dir is None or self.model in {"jev-latest", "jev-preview"}:
            return None
        identity = f"jev-wiki-provider:{self.CACHE_VERSION}:{self.ENDPOINT}:".encode() + payload
        digest = hashlib.sha256(identity).hexdigest()
        return self.cache_dir / f"v{self.CACHE_VERSION}-{digest}.json"

    def _read_cache(self, path: Path) -> dict:
        if path.parent.is_symlink():
            raise ProviderError("Cache directory must not be a symlink")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RESPONSE_BYTES:
                raise ProviderError("Cached response must be a bounded regular file")
            data = handle.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise ProviderError("Cached response exceeds size limit")
        return json.loads(data)

    def _write_cache(self, path: Path, raw: dict) -> None:
        temporary = None
        try:
            if path.parent.is_symlink():
                raise ProviderError("Cache directory must not be a symlink")
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, temporary = tempfile.mkstemp(prefix=".jev-", dir=path.parent)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_json_bytes(raw))
                handle.flush()
            os.replace(temporary, path)
        except (OSError, ProviderError):
            self.telemetry["cache_errors"] += 1
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _request(self, encoded: bytes) -> dict:
        for attempt in range(self.MAX_ATTEMPTS):
            request = urllib.request.Request(
                self.ENDPOINT,
                data=encoded,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            self.telemetry["requests"] += 1
            try:
                response = self._transport(request, self.timeout)
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                ConnectionError,
                OSError,
            ):
                if attempt == self.MAX_ATTEMPTS - 1:
                    raise ProviderError(
                        "JEV network request failed after bounded retries"
                    ) from None
                self._backoff(attempt, {})
                continue
            if response.status == 200:
                if len(response.body) > MAX_RESPONSE_BYTES:
                    raise ProviderError("JEV response exceeded the response-size limit")
                try:
                    return json.loads(response.body)
                except (ValueError, UnicodeError):
                    raise ProviderError("JEV returned invalid JSON") from None
            if response.status not in self.RETRY_STATUSES or attempt == self.MAX_ATTEMPTS - 1:
                # Bodies can contain source text or echoed headers: do not expose them.
                raise ProviderError(f"JEV HTTP {response.status}; no complete decisions available")
            self._backoff(attempt, response.headers)
        raise ProviderError("JEV request exhausted its attempt budget")

    def _backoff(self, attempt: int, headers: Mapping[str, str]) -> None:
        delay = 0.5 * (2**attempt)
        retry_after = next(
            (value for key, value in headers.items() if key.lower() == "retry-after"), None
        )
        if retry_after is not None:
            try:
                seconds = float(retry_after)
                if math.isfinite(seconds) and seconds >= 0:
                    delay = max(delay, min(seconds, 5.0))
            except (TypeError, ValueError):
                pass
        self.telemetry["retries"] += 1
        self._sleep(delay)


class ScriptedProvider:
    """Explicit local test double. Each ask consumes one pre-authored answer set."""

    def __init__(self, responses: list[dict[str, dict]]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, dict]]] = []

    def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        self.calls.append((state, questions))
        if not self._responses:
            raise ProviderError("ScriptedProvider has no remaining scripted responses")
        answers = self._responses.pop(0)
        if set(answers) != set(questions):
            raise ProviderError("ScriptedProvider answer identifiers do not match questions")
        return answers
