"""One live request to the TypeSafe API; runs only with ``RUN_LIVE_TESTS=1``."""

import os
from typing import TYPE_CHECKING

import pytest

from system_one_datasets.client import SystemOneClient, answer_distribution


if TYPE_CHECKING:
    from system_one_datasets.schema import JSONValue


API_KEY_ENV = "TYPESAFE_API_KEY"

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to run")


def test_live_jev_single_request_shape() -> None:
    """One request with one question of each type against jev-1.13.0; asserts response shape only."""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        pytest.fail(f"RUN_LIVE_TESTS=1 but {API_KEY_ENV} is not set")
    questions: dict[str, dict[str, JSONValue]] = {
        "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments, refunds", "technical": "Bugs, outages"},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    }
    with SystemOneClient("https://api.typesafe.ai", "jev-1.13.0", api_key=api_key, timeout=60) as client:
        evaluation = client.evaluate("Help! My payouts have been failing for 3 days.", questions)

    assert evaluation.model.startswith("jev")
    assert isinstance(evaluation.usage.get("input_tokens"), int)
    assert isinstance(evaluation.usage.get("output_tokens"), int)
    assert evaluation.latency_s > 0
    answers = evaluation.answers
    assert set(answers) >= set(questions)

    noul = answer_distribution(answers["urgent"], "noul", ["0", "1"])
    assert 0.0 <= noul["1"] <= 1.0

    team = answers["team"]
    assert team["choice"] in {"billing", "technical"}
    assert isinstance(team["confidence"], float)
    assert sum(answer_distribution(team, "choice", ["billing", "technical"]).values()) == pytest.approx(1.0, abs=1e-3)

    frustration = answers["frustration"]
    assert isinstance(frustration["score"], int | float)
    assert isinstance(frustration["legend"], dict)
    assert set(frustration["legend"]) == {"0", "1", "2"}
    assert isinstance(frustration["confidence"], float)
    levels = answer_distribution(frustration, "score", ["0", "1", "2"])
    assert sum(levels.values()) == pytest.approx(1.0, abs=1e-3)
