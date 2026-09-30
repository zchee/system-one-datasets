"""Loader tests on rows shaped exactly like jev-bench (trimmed copies of real test-split rows)."""

import orjson
import pytest

from system_one_datasets.data import normalize_soft_label, record_from_row


CIVIL_ROW = {
    "id": "civil_comments/test/71007",
    "state": '"A large chunk of doctors and clinics in Alaska won\'t take Medicare."',
    "question": orjson.dumps({
        "type": "noul",
        "instructions": "Is this comment toxic?",
        "criteria": {"true": "Rude.", "false": "Civil."},
    }).decode(),
    "label": "0",
    "soft_label": "0.2",
}
HATE_ROW = {
    "id": "measuring_hate_speech/test/1262",
    "state": '"some comment"',
    "question": orjson.dumps({
        "type": "score",
        "instructions": "Does this comment contain hate speech?",
        "criteria": ["No hate speech.", "Borderline.", "Hate speech."],
    }).decode(),
    "label": "2",
    "soft_label": "[0.0, 0.25, 0.75]",
}
EMOTION_ROW = {
    "id": "go_emotions/test/1392",
    "state": '"I\'d worry about traffic more than the TSA."',
    "question": orjson.dumps({
        "type": "choice",
        "instructions": "Which emotion does the comment primarily express?",
        "criteria": {"fear": None, "nervousness": None, "neutral": None},
    }).decode(),
    "label": "nervousness",
    "soft_label": '{"fear": 0.25, "nervousness": 0.5, "neutral": 0.25}',
}
STSB_ROW = {
    "id": "stsb/test/214",
    "state": '{"sentence1": "Someone is slicing ribs.", "sentence2": "Someone is cutting meat."}',
    "question": orjson.dumps({
        "type": "score",
        "instructions": "How similar in meaning are `sentence1` and `sentence2`?",
        "criteria": ["Dissimilar.", "Same topic.", "Equivalent."],
    }).decode(),
    "label": "1",
    "soft_label": None,
}


def test_noul_row() -> None:
    """Noul: options "0"/"1", string state decoded, scalar soft label becomes {"0": 1-p, "1": p}."""
    record = record_from_row(CIVIL_ROW, "civil_comments")
    assert record.kind == "noul"
    assert record.options == ["0", "1"]
    assert record.state == "A large chunk of doctors and clinics in Alaska won't take Medicare."
    assert record.question == orjson.loads(CIVIL_ROW["question"])
    assert record.soft_label == {"0": pytest.approx(0.8), "1": 0.2}


def test_score_row_with_array_soft_label() -> None:
    """Score: options are level indices and the soft-label array is keyed by them."""
    record = record_from_row(HATE_ROW, "measuring_hate_speech")
    assert record.kind == "score"
    assert record.options == ["0", "1", "2"]
    assert record.label == "2"
    assert record.soft_label == {"0": 0.0, "1": 0.25, "2": 0.75}


def test_choice_row_keeps_criteria_order() -> None:
    """Choice: options follow criteria key order; the question (null criteria values included) is untouched."""
    record = record_from_row(EMOTION_ROW, "go_emotions")
    assert record.kind == "choice"
    assert record.options == ["fear", "nervousness", "neutral"]
    assert record.question["criteria"] == {"fear": None, "nervousness": None, "neutral": None}
    assert record.soft_label == {"fear": 0.25, "nervousness": 0.5, "neutral": 0.25}


def test_object_state_and_missing_soft_label() -> None:
    """An object state is decoded to a dict and a null soft label stays None."""
    record = record_from_row(STSB_ROW, "stsb")
    assert record.state == {"sentence1": "Someone is slicing ribs.", "sentence2": "Someone is cutting meat."}
    assert record.soft_label is None


def test_label_outside_options_is_rejected() -> None:
    """A label that is not an option is a data error."""
    with pytest.raises(ValueError, match="not an option"):
        record_from_row({**HATE_ROW, "label": "7"}, "measuring_hate_speech")


@pytest.mark.parametrize("raw", [None, "", "  ", "null"])
def test_absent_soft_label_forms(raw: str | None) -> None:
    """None, empty, and JSON null all mean "no soft label"."""
    assert normalize_soft_label(raw, "score", ["0", "1"]) is None


def test_soft_label_shape_mismatch() -> None:
    """A score array with the wrong length is rejected."""
    with pytest.raises(ValueError, match="one entry per level"):
        normalize_soft_label("[0.5, 0.5]", "score", ["0", "1", "2"])
