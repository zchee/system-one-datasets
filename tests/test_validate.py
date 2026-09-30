"""Structural validator tests on small in-memory rows; one live test behind ``RUN_LIVE_TESTS=1``.

Rows are written readable (``state``/``question``/``soft_label`` as values) and ``encode`` turns them into the
on-disk form, where those three fields are JSON-encoded strings.
"""

import copy
import os
from typing import TYPE_CHECKING

import httpx
import orjson
import pytest

from system_one_datasets.client import SystemOneClient
from system_one_datasets.rows import CANONICAL_FIELDS, JSON_TEXT_FIELDS, decode_row, dumps
from system_one_datasets.validate import (
    check_row,
    data_files,
    live_check,
    render_summary,
    validate_file,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from system_one_datasets.schema import JSONValue

    type Row = dict[str, JSONValue]
    type Mutation = Callable[[Row], None]


def base(kind: str, question: dict[str, JSONValue], options: list[JSONValue], label: str, soft: JSONValue) -> Row:
    """A readable row with keys in ``CANONICAL_FIELDS`` order."""
    return {
        "id": f"cfg/test/{kind}",
        "suite": "suite",
        "config": "cfg",
        "kind": kind,
        "state": "some state",
        "question": question,
        "options": options,
        "label": label,
        "soft_label": soft,
        "source": "owner/dataset",
        "source_revision": "0" * 40,
        "upstream": "owner/original",
        "license": "cc-by-4.0",
    }


NOUL = base(
    "noul",
    {"type": "noul", "instructions": "Is this comment toxic?", "criteria": {"true": "Rude.", "false": "Civil."}},
    ["0", "1"],
    "0",
    {"0": 0.8, "1": 0.2},
)
CHOICE = base(
    "choice",
    {"type": "choice", "instructions": "Which tool next?", "criteria": {"search": "Web search.", "finish": None}},
    ["search", "finish"],
    "finish",
    None,
)
SCORE = base(
    "score",
    {"type": "score", "instructions": "How helpful?", "criteria": ["Not.", "Somewhat.", "Very."]},
    ["0", "1", "2"],
    "2",
    {"0": 0.1, "1": 0.2, "2": 0.7},
)


def encode(row: Row) -> Row:
    """The on-disk form: ``state``, ``question``, and ``soft_label`` as JSON text."""
    return {name: dumps(value) if name in JSON_TEXT_FIELDS else value for name, value in row.items()}


def errors(row: Row) -> list[str]:
    """Messages of the error-severity issues of the encoded row."""
    return [i.message for i in check_row(encode(row)) if i.severity == "error"]


@pytest.mark.parametrize("row", [NOUL, CHOICE, SCORE], ids=["noul", "choice", "score"])
def test_valid_rows_have_no_issues(row: Row) -> None:
    """One well-formed row per kind passes with no errors and no warnings, and decodes to its readable form."""
    encoded = encode(row)
    assert tuple(encoded) == CANONICAL_FIELDS
    assert all(isinstance(encoded[name], str) for name in JSON_TEXT_FIELDS)
    assert check_row(encoded) == []
    record = decode_row(encoded)
    assert (record.state, record.question, record.soft_label) == (row["state"], row["question"], row["soft_label"])


def mutate(row: Row, fn: Mutation) -> Row:
    """Deep-copy ``row`` and apply ``fn`` to the copy."""
    out = copy.deepcopy(row)
    fn(out)
    return out


def set_question(key: str, value: JSONValue) -> Mutation:
    """Mutation replacing ``question[key]``."""

    def fn(row: Row) -> None:
        question = row["question"]
        assert isinstance(question, dict)
        question[key] = value

    return fn


def set_criteria(value: JSONValue) -> Mutation:
    """Mutation replacing ``question.criteria``."""
    return set_question("criteria", value)


def set_field(name: str, value: JSONValue) -> Mutation:
    """Mutation replacing a top-level field."""

    def fn(row: Row) -> None:
        row[name] = value

    return fn


def drop_field(name: str) -> Mutation:
    """Mutation deleting a top-level field."""

    def fn(row: Row) -> None:
        del row[name]

    return fn


MANY: JSONValue = {f"o{i}": None for i in range(256)}

FAILURES: dict[str, tuple[Row, Mutation, str]] = {
    "error: missing field": (NOUL, drop_field("license"), "missing fields"),
    "error: unknown kind": (NOUL, set_field("kind", "rank"), "unknown kind"),
    "error: question type differs from kind": (SCORE, set_field("kind", "choice"), "!= kind"),
    "error: empty instructions": (NOUL, set_question("instructions", ""), "instructions"),
    "error: null state": (NOUL, set_field("state", None), "state must be"),
    "error: noul options not 0/1": (NOUL, set_field("options", ["no", "yes"]), "noul options"),
    "error: label not an option": (NOUL, set_field("label", "2"), "not an option"),
    "error: choice criteria not an object": (CHOICE, set_criteria(["search", "finish"]), "must be an object"),
    "error: choice criteria single entry": (CHOICE, set_criteria({"search": None}), "want 2..255"),
    "error: choice criteria over 255 entries": (CHOICE, set_criteria(MANY), "want 2..255"),
    "error: choice options differ from criteria keys": (
        CHOICE,
        set_field("options", ["finish", "search"]),
        "equal the criteria keys",
    ),
    "error: choice criteria value not a string": (
        CHOICE,
        set_criteria({"search": 1, "finish": None}),
        "strings or null",
    ),
    "error: score single level": (SCORE, set_criteria(["Only."]), "want 2..10"),
    "error: score eleven levels": (SCORE, set_criteria([str(i) for i in range(11)]), "want 2..10"),
    "error: score options not level indices": (SCORE, set_field("options", ["1", "2", "3"]), "level indices"),
    "error: score label outside levels": (SCORE, set_field("label", "3"), "not an option"),
    "error: soft_label key not an option": (NOUL, set_field("soft_label", {"0": 0.5, "x": 0.5}), "not in options"),
    "error: soft_label sum too low": (SCORE, set_field("soft_label", {"0": 0.1, "1": 0.1, "2": 0.1}), "sums to"),
    "error: soft_label sum too high": (NOUL, set_field("soft_label", {"0": 0.6, "1": 0.6}), "sums to"),
    "error: soft_label negative": (NOUL, set_field("soft_label", {"0": 1.5, "1": -0.5}), "in [0, 1]"),
    "error: soft_label not an object": (NOUL, set_field("soft_label", [0.8, 0.2]), "non-empty object"),
    "error: empty source_revision": (CHOICE, set_field("source_revision", ""), "source_revision"),
    "error: empty upstream": (CHOICE, set_field("upstream", ""), "upstream must be a non-empty string"),
    "error: missing upstream": (SCORE, drop_field("upstream"), "missing fields: ['upstream']"),
}


@pytest.mark.parametrize(("row", "fn", "expected"), list(FAILURES.values()), ids=list(FAILURES))
def test_failure_modes(row: Row, fn: Mutation, expected: str) -> None:
    """Each malformed row yields an error whose message names the problem."""
    messages = errors(mutate(row, fn))
    assert any(expected in m for m in messages), messages


RAW_FAILURES: dict[str, tuple[str, JSONValue, str]] = {
    "error: state stored as a nested object": ("state", {"a": 1}, "state must be a JSON-encoded string"),
    "error: question text is not JSON": ("question", "{not json", "question is not valid JSON text"),
    "error: absent soft_label stored as JSON null instead of the text null": (
        "soft_label",
        None,
        "soft_label must be a JSON-encoded string",
    ),
}


@pytest.mark.parametrize(("name", "value", "expected"), list(RAW_FAILURES.values()), ids=list(RAW_FAILURES))
def test_json_text_fields_must_be_strings(name: str, value: JSONValue, expected: str) -> None:
    """The three JSON-text fields must be strings holding valid JSON."""
    row = {**encode(CHOICE), name: value}
    assert [i.message for i in check_row(row)] == [expected]


def test_soft_label_sum_tolerance() -> None:
    """Sums inside [0.98, 1.02] pass; the boundary is inclusive."""
    assert errors({**NOUL, "soft_label": {"0": 0.51, "1": 0.51}}) == []
    assert errors({**NOUL, "soft_label": {"0": 0.49, "1": 0.49}}) == []


def test_non_object_row() -> None:
    """A JSON array is not a row."""
    assert [i.message for i in check_row([1, 2])] == ["row is not a JSON object"]


def test_large_state_is_a_warning_not_an_error() -> None:
    """~32k+ estimated tokens (chars/3) warns without failing the row."""
    issues = check_row(encode({**CHOICE, "state": "x" * 100_000}))
    assert [i.severity for i in issues] == ["warning"]
    assert "tokens" in issues[0].message


def write(path: Path, rows: list[Row], extra: bytes = b"") -> Path:
    """Write encoded rows as JSONL, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(orjson.dumps(encode(r)) + b"\n" for r in rows) + extra)
    return path


def test_validate_file_counts_and_file_level_checks(tmp_path: Path) -> None:
    """Directory names must match suite/config; duplicate ids and bad JSON lines are errors."""
    good = write(tmp_path / "suite" / "cfg" / "test.jsonl", [NOUL, CHOICE, SCORE])
    result = validate_file(good)
    assert (result.rows, result.errors, result.warnings) == (3, 0, 0)
    assert dict(result.kinds) == {"noul": 1, "choice": 1, "score": 1}

    bad = write(tmp_path / "other" / "cfg" / "validation.jsonl", [NOUL, NOUL], extra=b"{not json\n")
    result = validate_file(bad)
    assert result.rows == 3
    joined = "\n".join(result.messages)
    assert "suite 'suite' != directory 'other'" in joined
    assert "duplicate id 'cfg/test/noul'" in joined
    assert "invalid JSON" in joined
    # Two suite mismatches, one duplicate id, one invalid line.
    assert result.errors == 4


def test_empty_file_is_an_error(tmp_path: Path) -> None:
    """A file with no rows fails."""
    empty = write(tmp_path / "suite" / "cfg" / "test.jsonl", [])
    assert validate_file(empty).errors == 1


def test_data_files_and_summary(tmp_path: Path) -> None:
    """Only <suite>/<config>/<split>.jsonl files are found; the summary has one line per file."""
    write(tmp_path / "suite" / "cfg" / "test.jsonl", [NOUL])
    write(tmp_path / "suite" / "cfg" / "validation.jsonl", [{**NOUL, "id": "cfg/validation/0"}])
    (tmp_path / "manifest.yaml").write_text("name: x\n", encoding="utf-8")
    files = data_files(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in files] == ["suite/cfg/test.jsonl", "suite/cfg/validation.jsonl"]
    summary = render_summary([validate_file(p) for p in files])
    lines = summary.splitlines()
    assert lines[0].split() == ["suite/config", "split", "rows", "kinds", "errors", "warnings"]
    assert lines[1].split() == ["suite/cfg", "test", "1", "noul:1", "0", "0"]


def mock_backend(answer_for: Callable[[Row], Row], status: int = 200) -> SystemOneClient:
    """Client whose backend answers each request's single question with ``answer_for(question)``."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = orjson.loads(request.content)
        if status != 200:
            return httpx.Response(status, text="boom")
        answers = {qid: answer_for(q) for qid, q in body["questions"].items()}
        return httpx.Response(200, content=orjson.dumps({"model": "m", "answers": answers, "usage": {}}))

    return SystemOneClient("https://example.test", "m", transport=httpx.MockTransport(handler), max_retries=0)


def kind_answer(question: Row) -> Row:
    """A well-formed answer of the question's own type."""
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        return {"type": "noul", "noul": 0.3}
    levels = criteria if isinstance(criteria, list) else []
    keys = list(criteria) if isinstance(criteria, dict) else [str(i) for i in range(len(levels))]
    probabilities: Row = {key: 1 / len(keys) for key in keys}
    return {"type": kind, "probabilities": probabilities}


def test_live_check_passes_matching_answers() -> None:
    """HTTP 200 with an answer of the question's kind passes, for every kind."""
    with mock_backend(kind_answer) as client:
        result = live_check("suite/cfg", [decode_row(encode(r)) for r in (NOUL, CHOICE, SCORE)], client)
    assert (result.sent, result.passed, result.failures) == (3, 3, [])


def test_live_check_reports_kind_mismatch_and_http_errors() -> None:
    """A noul answer to a choice question fails, as does a non-200 status."""
    with mock_backend(lambda _q: {"type": "noul", "noul": 0.5}) as client:
        result = live_check("suite/cfg", [decode_row(encode(CHOICE))], client)
    assert result.passed == 0
    assert "answer type 'noul' != question type 'choice'" in result.failures[0]

    with mock_backend(kind_answer, status=400) as client:
        result = live_check("suite/cfg", [decode_row(encode(NOUL))], client)
    assert result.passed == 0
    assert "HTTP 400" in result.failures[0]


@pytest.mark.skipif(os.environ.get("RUN_LOCAL_LIVE_TESTS") != "1", reason="set RUN_LOCAL_LIVE_TESTS=1 to run")
def test_live_local_backend_answers_every_kind() -> None:
    """One row per kind against a running /v1/systemone backend (``SYSTEMONE_BASE_URL``, default localhost:3000)."""
    base_url = os.environ.get("SYSTEMONE_BASE_URL", "http://localhost:3000")
    model = os.environ.get("SYSTEMONE_MODEL", "openjev")
    with SystemOneClient(base_url, model, timeout=300, max_retries=1) as client:
        result = live_check("suite/cfg", [decode_row(encode(r)) for r in (NOUL, CHOICE, SCORE)], client)
    assert result.failures == []
    assert result.passed == 3
