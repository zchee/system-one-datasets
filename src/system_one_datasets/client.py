"""HTTP client for the ``/v1/systemone`` evaluation endpoint.

Wire format: https://docs.typesafe.ai/api.md.
"""

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import logging
import random
import time
from typing import TYPE_CHECKING, NamedTuple, Self

import httpx
import orjson

from system_one_datasets.schema import NOUL_OPTIONS, JSONValue, Kind


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import TracebackType


logger = logging.getLogger(__name__)

ENDPOINT = "/v1/systemone"
_ERROR_BODY_LIMIT = 500


class SystemOneError(Exception):
    """A request the backend answered with a non-retryable or exhausted-retry error status."""

    def __init__(self, status: int, body: str) -> None:
        """Store the HTTP status and a truncated response body (never request headers).

        Args:
            status: HTTP status code of the final response.
            body: Response body text.
        """
        self.status = status
        self.body = body[:_ERROR_BODY_LIMIT]
        super().__init__(f"HTTP {status}: {self.body}")


class AnswerParseError(ValueError):
    """The response did not contain a well-formed answer for a question."""


class Evaluation(NamedTuple):
    """Result of one successful ``evaluate`` call.

    Attributes:
        answers: Answer objects keyed by question id, as returned by the backend.
        usage: Token usage (``input_tokens``, ``output_tokens``).
        latency_s: Wall-clock seconds of the successful HTTP attempt; retries and back-off sleeps are excluded.
        model: Model name reported by the backend.
    """

    answers: dict[str, dict[str, JSONValue]]
    usage: dict[str, int]
    latency_s: float
    model: str


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header value into seconds.

    Args:
        value: Header value: delay in seconds, or an HTTP-date.
        now: Reference time for HTTP-date values; defaults to the current UTC time.

    Returns:
        Non-negative delay in seconds, or ``None`` when the header is missing or unparsable.
    """
    if not value:
        return None
    value = value.strip()
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - (now or datetime.now(UTC))).total_seconds(), 0.0)


class SystemOneClient:
    """Synchronous client for a backend that speaks the TypeSafe ``/v1/systemone`` wire format.

    Retries 429 (honoring ``Retry-After``), 5xx (including 529 Overloaded), and transport errors with
    exponential back-off plus jitter. Other 4xx statuses raise immediately.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60,
        *,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        max_backoff: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a client.

        Args:
            base_url: Backend origin, e.g. ``https://api.typesafe.ai``.
            model: Model name sent in every request.
            api_key: Bearer token; ``None`` sends no ``Authorization`` header.
            timeout: Per-attempt HTTP timeout in seconds.
            max_retries: Retries after the first attempt before giving up.
            backoff_base: First back-off delay in seconds; doubles on each retry.
            max_backoff: Upper bound for any single sleep, including ``Retry-After``.
            transport: Optional httpx transport (e.g. ``httpx.MockTransport`` in tests).
            sleep: Sleep function, injectable so tests do not wait.
        """
        self.model = model
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._max_backoff = max_backoff
        self._sleep = sleep
        headers = {"content-type": "application/json", "accept": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=timeout, transport=transport)

    def __repr__(self) -> str:
        """Describe the client without exposing credentials."""
        return f"SystemOneClient(base_url={str(self._http.base_url)!r}, model={self.model!r})"

    def __enter__(self) -> Self:
        """Enter a context that closes the HTTP connection pool on exit."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Close the HTTP connection pool."""
        self.close()

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._http.close()

    def _backoff(self, attempt: int) -> float:
        delay = self._backoff_base * (2**attempt)
        return min(delay + random.uniform(0, delay / 2), self._max_backoff)  # Jitter, not crypto.

    def evaluate(self, state: JSONValue, questions: Mapping[str, Mapping[str, JSONValue]]) -> Evaluation:
        """Evaluate ``state`` against ``questions``.

        Args:
            state: String, object, or array sent as the request ``state``.
            questions: Wire-format Question objects keyed by caller-chosen ids.

        Returns:
            The parsed answers, usage, latency of the successful attempt, and the model the backend reported.

        Raises:
            SystemOneError: Non-retryable status, or retryable status after retries are exhausted.
            httpx.TransportError: Network failure after retries are exhausted.
            AnswerParseError: The response body is malformed or lacks an answer for a question.
        """
        body = orjson.dumps({"state": state, "model": self.model, "questions": questions})
        attempt = 0
        while True:
            started = time.perf_counter()
            try:
                response = self._http.post(ENDPOINT, content=body)
            except httpx.TransportError as exc:
                if attempt >= self._max_retries:
                    raise
                delay = self._backoff(attempt)
                logger.warning("transport error %s; retry %d in %.2fs", type(exc).__name__, attempt + 1, delay)
            else:
                latency = time.perf_counter() - started
                status = response.status_code
                if status == httpx.codes.OK:
                    return _parse_response(response.content, questions, latency)
                retryable = status == httpx.codes.TOO_MANY_REQUESTS or status >= httpx.codes.INTERNAL_SERVER_ERROR
                if not retryable or attempt >= self._max_retries:
                    raise SystemOneError(status, response.text)
                delay = self._backoff(attempt)
                if status == httpx.codes.TOO_MANY_REQUESTS:
                    retry_after = parse_retry_after(response.headers.get("retry-after"))
                    if retry_after is not None:
                        delay = min(retry_after, self._max_backoff)
                logger.warning("HTTP %d; retry %d in %.2fs", status, attempt + 1, delay)
            self._sleep(delay)
            attempt += 1


def _parse_response(content: bytes, questions: Mapping[str, object], latency: float) -> Evaluation:
    try:
        payload = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        msg = "response is not JSON"
        raise AnswerParseError(msg) from exc
    if not isinstance(payload, dict) or not isinstance(answers := payload.get("answers"), dict):
        msg = "response has no answers object"
        raise AnswerParseError(msg)
    missing = [qid for qid in questions if not isinstance(answers.get(qid), dict)]
    if missing:
        msg = f"missing answers for {missing}"
        raise AnswerParseError(msg)
    raw_usage = payload.get("usage")
    usage = (
        {k: int(v) for k, v in raw_usage.items() if isinstance(v, int | float)} if isinstance(raw_usage, dict) else {}
    )
    return Evaluation(answers=answers, usage=usage, latency_s=latency, model=str(payload.get("model", "")))


def answer_distribution(answer: Mapping[str, JSONValue], kind: Kind, options: Sequence[str]) -> dict[str, float]:
    """Convert one wire-format Answer into a full probability distribution over ``options``.

    Noul answers carry ``noul`` = P(yes), mapped to ``{"0": 1 - p, "1": p}``. Choice and score answers carry
    ``probabilities`` keyed by option / level-index string; options the backend omitted get 0.0.

    Args:
        answer: Answer object (``{"type": ..., ...}``).
        kind: Expected question kind.
        options: Ordered option keys of the question.

    Returns:
        Mapping of every option in ``options`` order to its probability.

    Raises:
        AnswerParseError: The answer type does not match ``kind`` or required fields are missing.
    """
    if answer.get("type") != kind:
        msg = f"answer type {answer.get('type')!r} != question type {kind!r}"
        raise AnswerParseError(msg)
    if kind == "noul":
        p = answer.get("noul")
        if not isinstance(p, int | float) or isinstance(p, bool):
            msg = "noul answer lacks numeric 'noul'"
            raise AnswerParseError(msg)
        no, yes = NOUL_OPTIONS
        return {no: 1.0 - float(p), yes: float(p)}
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        msg = f"{kind} answer lacks 'probabilities'"
        raise AnswerParseError(msg)
    unknown = set(probabilities) - set(options)
    if unknown:
        msg = f"{kind} answer has unknown options {sorted(unknown)}"
        raise AnswerParseError(msg)
    dist: dict[str, float] = {}
    for option in options:
        value = probabilities.get(option, 0.0)
        if not isinstance(value, int | float) or isinstance(value, bool):
            msg = f"non-numeric probability for {option!r}"
            raise AnswerParseError(msg)
        dist[option] = float(value)
    return dist
