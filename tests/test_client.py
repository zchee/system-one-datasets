"""Client tests against ``httpx.MockTransport`` using the response bodies from docs.typesafe.ai/api.md."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import orjson
import pytest

from system_one_datasets.client import (
    AnswerParseError,
    SystemOneClient,
    SystemOneError,
    answer_distribution,
    parse_retry_after,
)


if TYPE_CHECKING:
    from system_one_datasets.schema import JSONValue, Kind


FAKE_KEY = "sk-test-not-a-real-key"
STATE = "Help! My payouts have been failing for 3 days."

NOUL_RESPONSE: dict[str, JSONValue] = {
    "model": "jev-1.13.0",
    "answers": {"is_urgent": {"type": "noul", "noul": 0.95}},
    "usage": {"input_tokens": 307, "output_tokens": 20},
}
CHOICE_RESPONSE: dict[str, JSONValue] = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
            "confidence": 0.81,
        }
    },
    "usage": {"input_tokens": 318, "output_tokens": 34},
}
SCORE_RESPONSE: dict[str, JSONValue] = {
    "model": "jev-1.13.0",
    "answers": {
        "frustration": {
            "type": "score",
            "score": 1.05,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
            "confidence": 0.92,
        }
    },
    "usage": {"input_tokens": 304, "output_tokens": 18},
}


class Recorder:
    """Mock transport handler that replays queued responses and records requests."""

    def __init__(self, *responses: httpx.Response) -> None:
        """Queue responses in the order they are returned."""
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Record the request and return the next queued response."""
        self.requests.append(request)
        return self.responses.pop(0)


def make_client(
    recorder: Recorder, sleeps: list[float], max_retries: int = 5, backoff_base: float = 1.0, max_backoff: float = 60.0
) -> SystemOneClient:
    """Build a client wired to ``recorder`` that records sleeps instead of sleeping."""
    return SystemOneClient(
        "https://api.typesafe.ai",
        "jev-1.13.0",
        api_key=FAKE_KEY,
        max_retries=max_retries,
        backoff_base=backoff_base,
        max_backoff=max_backoff,
        transport=httpx.MockTransport(recorder),
        sleep=sleeps.append,
    )


def ok(body: dict[str, JSONValue]) -> httpx.Response:
    """Build a 200 JSON response."""
    return httpx.Response(200, content=orjson.dumps(body))


@pytest.mark.parametrize(
    ("question_id", "question", "response", "kind", "options", "expected"),
    [
        pytest.param(
            "is_urgent",
            {"type": "noul", "instructions": "Does this convey urgency?"},
            NOUL_RESPONSE,
            "noul",
            ["0", "1"],
            {"0": pytest.approx(0.05), "1": 0.95},
            id="noul",
        ),
        pytest.param(
            "department",
            {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments", "technical": "Bugs", "sales": "Pricing"},
            },
            CHOICE_RESPONSE,
            "choice",
            ["billing", "technical", "sales"],
            {"billing": 0.88, "technical": 0.12, "sales": 0.0},
            id="choice",
        ),
        pytest.param(
            "frustration",
            {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
            SCORE_RESPONSE,
            "score",
            ["0", "1", "2"],
            {"0": 0.0, "1": 0.95, "2": 0.05},
            id="score",
        ),
    ],
)
def test_evaluate_parses_each_answer_type(
    question_id: str,
    question: dict[str, JSONValue],
    response: dict[str, JSONValue],
    kind: Kind,
    options: list[str],
    expected: dict[str, float],
) -> None:
    """The request carries state/model/questions verbatim and each answer type maps to a full distribution."""
    recorder = Recorder(ok(response))
    sleeps: list[float] = []
    with make_client(recorder, sleeps) as client:
        evaluation = client.evaluate(STATE, {question_id: question})

    (request,) = recorder.requests
    assert request.method == "POST"
    assert request.url == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert orjson.loads(request.content) == {
        "state": STATE,
        "model": "jev-1.13.0",
        "questions": {question_id: question},
    }
    assert evaluation.model == "jev-1.13.0"
    assert evaluation.usage == response["usage"]
    assert evaluation.latency_s >= 0
    assert sleeps == []
    dist = answer_distribution(evaluation.answers[question_id], kind, options)
    assert list(dist) == options
    assert dist == expected


def test_retry_after_seconds_is_honored_on_429() -> None:
    """A 429 with ``Retry-After: 2`` sleeps exactly 2 seconds, then the retry succeeds."""
    recorder = Recorder(httpx.Response(429, headers={"retry-after": "2"}, text="slow down"), ok(NOUL_RESPONSE))
    sleeps: list[float] = []
    with make_client(recorder, sleeps) as client:
        evaluation = client.evaluate(STATE, {"is_urgent": {"type": "noul", "instructions": "Urgent?"}})
    assert sleeps == [2.0]
    assert len(recorder.requests) == 2
    assert evaluation.answers["is_urgent"]["noul"] == pytest.approx(0.95)


def test_retry_after_is_capped_by_max_backoff() -> None:
    """A huge ``Retry-After`` is clamped to ``max_backoff``."""
    recorder = Recorder(httpx.Response(429, headers={"retry-after": "3600"}), ok(NOUL_RESPONSE))
    sleeps: list[float] = []
    with make_client(recorder, sleeps, max_backoff=30.0) as client:
        client.evaluate(STATE, {"is_urgent": {"type": "noul", "instructions": "Urgent?"}})
    assert sleeps == [30.0]


def test_429_without_retry_after_uses_backoff() -> None:
    """Without ``Retry-After`` the first delay is the back-off base plus at most 50% jitter."""
    recorder = Recorder(httpx.Response(429), ok(NOUL_RESPONSE))
    sleeps: list[float] = []
    with make_client(recorder, sleeps, backoff_base=1.0) as client:
        client.evaluate(STATE, {"is_urgent": {"type": "noul", "instructions": "Urgent?"}})
    assert len(sleeps) == 1
    assert 1.0 <= sleeps[0] <= 1.5


def test_5xx_retries_with_exponential_backoff() -> None:
    """529 and 503 are retried with doubling delays (base 1s, then 2s, each with up to 50% jitter)."""
    recorder = Recorder(httpx.Response(529), httpx.Response(503), ok(NOUL_RESPONSE))
    sleeps: list[float] = []
    with make_client(recorder, sleeps, backoff_base=1.0) as client:
        client.evaluate(STATE, {"is_urgent": {"type": "noul", "instructions": "Urgent?"}})
    assert len(recorder.requests) == 3
    assert 1.0 <= sleeps[0] <= 1.5
    assert 2.0 <= sleeps[1] <= 3.0


def test_5xx_gives_up_after_max_retries() -> None:
    """After ``max_retries`` retries the last 5xx is raised as ``SystemOneError``."""
    recorder = Recorder(*[httpx.Response(503, text="unavailable")] * 3)
    sleeps: list[float] = []
    with make_client(recorder, sleeps, max_retries=2) as client, pytest.raises(SystemOneError) as info:
        client.evaluate(STATE, {"is_urgent": {"type": "noul", "instructions": "Urgent?"}})
    assert info.value.status == 503
    assert len(recorder.requests) == 3
    assert len(sleeps) == 2


def test_transport_error_is_retried() -> None:
    """A connection error is retried like a 5xx."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("refused", request=request)
        return ok(NOUL_RESPONSE)

    sleeps: list[float] = []
    client = SystemOneClient(
        "http://localhost:3000", "openjev", transport=httpx.MockTransport(handler), sleep=sleeps.append
    )
    with client:
        evaluation = client.evaluate(STATE, {"is_urgent": {"type": "noul", "instructions": "Urgent?"}})
    assert calls == 2
    assert len(sleeps) == 1
    assert evaluation.answers["is_urgent"]["noul"] == pytest.approx(0.95)


def test_422_is_not_retried_and_does_not_leak_the_key() -> None:
    """Validation errors raise immediately; neither the error nor ``repr`` exposes the API key."""
    recorder = Recorder(httpx.Response(422, text='{"detail": "questions.x.criteria: field required"}'))
    sleeps: list[float] = []
    client = make_client(recorder, sleeps)
    with client, pytest.raises(SystemOneError) as info:
        client.evaluate(STATE, {"x": {"type": "choice", "instructions": "Pick"}})
    assert info.value.status == 422
    assert "criteria" in str(info.value)
    assert sleeps == []
    assert FAKE_KEY not in str(info.value)
    assert FAKE_KEY not in repr(client)


def test_no_authorization_header_without_key() -> None:
    """The local backend sends no Authorization header."""
    recorder = Recorder(ok(NOUL_RESPONSE))
    client = SystemOneClient("http://localhost:3000", "openjev", transport=httpx.MockTransport(recorder))
    with client:
        client.evaluate(STATE, {"is_urgent": {"type": "noul", "instructions": "Urgent?"}})
    assert "authorization" not in recorder.requests[0].headers
    assert recorder.requests[0].url == "http://localhost:3000/v1/systemone"


def test_missing_answer_raises_parse_error() -> None:
    """A 200 without the requested question id is a parse error, not a silent empty answer."""
    recorder = Recorder(ok({"model": "m", "answers": {}, "usage": {}}))
    with make_client(recorder, []) as client, pytest.raises(AnswerParseError):
        client.evaluate(STATE, {"is_urgent": {"type": "noul", "instructions": "Urgent?"}})


def test_parse_retry_after_forms() -> None:
    """Delta-seconds, HTTP-date, past dates, and garbage are all handled."""
    now = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("7") == pytest.approx(7.0)
    assert parse_retry_after(" 1.5 ") == pytest.approx(1.5)
    assert parse_retry_after("Wed, 30 Sep 2026 12:00:10 GMT", now=now) == pytest.approx(10.0)
    assert parse_retry_after("Wed, 30 Sep 2026 11:00:00 GMT", now=now) == pytest.approx(0.0)
    assert parse_retry_after("-3") == pytest.approx(0.0)
    assert parse_retry_after("soon") is None
    assert parse_retry_after(None) is None


def test_answer_distribution_fills_missing_options_and_checks_type() -> None:
    """Options omitted by the backend get 0.0; a type mismatch or unknown option is rejected."""
    answer: dict[str, JSONValue] = {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "c": 0.4}}
    assert answer_distribution(answer, "choice", ["a", "b", "c"]) == {"a": 0.6, "b": 0.0, "c": 0.4}
    with pytest.raises(AnswerParseError, match="type"):
        answer_distribution(answer, "score", ["0", "1"])
    with pytest.raises(AnswerParseError, match="unknown options"):
        answer_distribution(answer, "choice", ["a", "b"])
    with pytest.raises(AnswerParseError, match="noul"):
        answer_distribution({"type": "noul"}, "noul", ["0", "1"])
