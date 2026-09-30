"""Structural and live validation of materialized dataset files (``data/<suite>/<config>/<split>.jsonl``)."""

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Literal

import httpx
import orjson

from system_one_datasets.client import AnswerParseError, SystemOneError, answer_distribution
from system_one_datasets.rows import CANONICAL_FIELDS, JSON_TEXT_FIELDS, decode_row, read_rows
from system_one_datasets.schema import KINDS, NOUL_OPTIONS


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from system_one_datasets.client import SystemOneClient
    from system_one_datasets.schema import BenchRecord, JSONValue


CHOICE_MIN_OPTIONS = 2
CHOICE_MAX_OPTIONS = 255
SCORE_MIN_LEVELS = 2
SCORE_MAX_LEVELS = 10
SOFT_LABEL_SUM_RANGE = (0.98, 1.02)
CHARS_PER_TOKEN = 3
TOKEN_WARN_LIMIT = 32_000
LIVE_SAMPLE_ROWS = 3
LIVE_QUESTION_ID = "q"

type Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class Issue:
    """One problem found in a row.

    Attributes:
        severity: ``error`` fails validation; ``warning`` is reported only.
        message: Human-readable description.
    """

    severity: Severity
    message: str


@dataclass(slots=True)
class FileResult:
    """Validation outcome for one ``<split>.jsonl`` file.

    Attributes:
        path: File that was checked.
        suite: Suite directory name.
        config: Config directory name.
        split: File stem.
        rows: Number of non-empty lines.
        kinds: Row count per question kind.
        errors: Number of error issues.
        warnings: Number of warning issues.
        messages: The first issues as ``"<line>: <message>"`` strings, capped to keep output readable.
    """

    path: Path
    suite: str
    config: str
    split: str
    rows: int = 0
    kinds: Counter[str] = field(default_factory=Counter)
    errors: int = 0
    warnings: int = 0
    messages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LiveResult:
    """Outcome of the live sample for one config.

    Attributes:
        config_key: ``<suite>/<config>``.
        sent: Rows posted.
        passed: Rows answered with HTTP 200 and an answer of the question's kind.
        failures: One message per failed row.
    """

    config_key: str
    sent: int = 0
    passed: int = 0
    failures: list[str] = field(default_factory=list)


_MAX_MESSAGES = 20


def estimate_tokens(state: JSONValue, question: JSONValue) -> int:
    """Rough token estimate of a request's ``state`` plus ``question``: serialized JSON characters / 3.

    Args:
        state: Request state.
        question: Wire-format Question object.

    Returns:
        Estimated token count.
    """
    chars = len(orjson.dumps(state).decode()) + len(orjson.dumps(question).decode())
    return chars // CHARS_PER_TOKEN


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _check_choice(question: Mapping[str, JSONValue], options: list[str]) -> list[Issue]:
    criteria = question.get("criteria")
    if not isinstance(criteria, dict):
        return [Issue("error", "choice criteria must be an object")]
    issues: list[Issue] = []
    if not CHOICE_MIN_OPTIONS <= len(criteria) <= CHOICE_MAX_OPTIONS:
        issues.append(
            Issue(
                "error", f"choice criteria has {len(criteria)} entries, want {CHOICE_MIN_OPTIONS}..{CHOICE_MAX_OPTIONS}"
            )
        )
    if list(criteria) != options:
        issues.append(Issue("error", "choice options must equal the criteria keys in order"))
    if any(value is not None and not isinstance(value, str) for value in criteria.values()):
        issues.append(Issue("error", "choice criteria values must be strings or null"))
    return issues


def _check_score(question: Mapping[str, JSONValue], options: list[str]) -> list[Issue]:
    criteria = question.get("criteria")
    if not isinstance(criteria, list):
        return [Issue("error", "score criteria must be an array")]
    issues: list[Issue] = []
    if not SCORE_MIN_LEVELS <= len(criteria) <= SCORE_MAX_LEVELS:
        issues.append(
            Issue("error", f"score criteria has {len(criteria)} levels, want {SCORE_MIN_LEVELS}..{SCORE_MAX_LEVELS}")
        )
    if not all(isinstance(level, str) for level in criteria):
        issues.append(Issue("error", "score criteria levels must be strings"))
    if options != [str(i) for i in range(len(criteria))]:
        issues.append(Issue("error", "score options must be the level indices '0'..'K-1'"))
    return issues


def _check_soft_label(soft_label: object, options: list[str]) -> list[Issue]:
    if soft_label is None:
        return []
    if not isinstance(soft_label, dict) or not soft_label:
        return [Issue("error", "soft_label must be a non-empty object or null")]
    issues: list[Issue] = []
    unknown = sorted(set(soft_label) - set(options))
    if unknown:
        issues.append(Issue("error", f"soft_label keys not in options: {unknown}"))
    values = list(soft_label.values())
    if not all(_is_number(v) and 0.0 <= v <= 1.0 for v in values):
        issues.append(Issue("error", "soft_label values must be numbers in [0, 1]"))
        return issues
    total = math.fsum(values)
    low, high = SOFT_LABEL_SUM_RANGE
    if not low <= total <= high:
        issues.append(Issue("error", f"soft_label sums to {total:.4f}, want [{low}, {high}]"))
    return issues


def check_row(row: object) -> list[Issue]:
    """Check one on-disk row: field types, then the decoded ``state``/``question``/``soft_label`` JSON text.

    Args:
        row: Decoded JSON value of one line.

    Returns:
        Every issue found; empty when the row is valid.
    """
    if not isinstance(row, dict):
        return [Issue("error", "row is not a JSON object")]
    missing = [name for name in CANONICAL_FIELDS if name not in row]
    if missing:
        return [Issue("error", f"missing fields: {missing}")]
    issues: list[Issue] = [
        Issue("error", f"{name} must be a non-empty string")
        for name in ("id", "suite", "config", "source", "source_revision", "upstream", "license")
        if not isinstance(row[name], str) or not row[name]
    ]
    decoded: dict[str, JSONValue] = {}
    for name in JSON_TEXT_FIELDS:
        text = row[name]
        if not isinstance(text, str):
            issues.append(Issue("error", f"{name} must be a JSON-encoded string"))
            continue
        try:
            decoded[name] = orjson.loads(text)
        except orjson.JSONDecodeError:
            issues.append(Issue("error", f"{name} is not valid JSON text"))
    if len(decoded) != len(JSON_TEXT_FIELDS):
        return issues
    state, question, soft_label = decoded["state"], decoded["question"], decoded["soft_label"]
    kind, options, label = row["kind"], row["options"], row["label"]
    if kind not in KINDS:
        return [*issues, Issue("error", f"unknown kind {kind!r}")]
    if not isinstance(state, str | dict | list):
        issues.append(Issue("error", "state must be a string, object, or array"))
    if not isinstance(question, dict):
        return [*issues, Issue("error", "question must be an object")]
    if question.get("type") != kind:
        issues.append(Issue("error", f"question type {question.get('type')!r} != kind {kind!r}"))
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        issues.append(Issue("error", "question instructions must be a non-empty string"))
    if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
        return [*issues, Issue("error", "options must be an array of strings")]
    if len(set(options)) != len(options):
        issues.append(Issue("error", "options must be unique"))
    if kind == "choice":
        issues += _check_choice(question, options)
    elif kind == "score":
        issues += _check_score(question, options)
    elif options != list(NOUL_OPTIONS):
        issues.append(Issue("error", f"noul options must be {list(NOUL_OPTIONS)}"))
    if not isinstance(label, str) or label not in options:
        issues.append(Issue("error", f"label {label!r} is not an option"))
    issues += _check_soft_label(soft_label, options)
    tokens = estimate_tokens(state, question)
    if tokens > TOKEN_WARN_LIMIT:
        issues.append(
            Issue("warning", f"state+question is ~{tokens} tokens (chars/{CHARS_PER_TOKEN}) > {TOKEN_WARN_LIMIT}")
        )
    return issues


def data_files(root: Path) -> list[Path]:
    """List ``<suite>/<config>/<split>.jsonl`` files under ``root`` in sorted order.

    Args:
        root: Dataset root directory.

    Returns:
        Sorted file paths.
    """
    return sorted(p for p in root.glob("*/*/*.jsonl") if p.is_file())


def validate_file(path: Path) -> FileResult:
    """Run ``check_row`` on every line of one file, plus per-file checks (duplicate ids, directory names).

    Args:
        path: ``<suite>/<config>/<split>.jsonl`` file.

    Returns:
        Counts and the first issues found.
    """
    result = FileResult(path=path, suite=path.parent.parent.name, config=path.parent.name, split=path.stem)
    seen: set[str] = set()

    def add(line_no: int, issue: Issue) -> None:
        if issue.severity == "error":
            result.errors += 1
        else:
            result.warnings += 1
        if len(result.messages) < _MAX_MESSAGES:
            result.messages.append(f"{path}:{line_no}: {issue.severity}: {issue.message}")

    with path.open("rb") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            result.rows += 1
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                add(line_no, Issue("error", f"invalid JSON: {exc}"))
                continue
            issues = check_row(row)
            if isinstance(row, dict):
                result.kinds[str(row.get("kind"))] += 1
                row_id = row.get("id")
                if isinstance(row_id, str):
                    if row_id in seen:
                        issues.append(Issue("error", f"duplicate id {row_id!r}"))
                    seen.add(row_id)
                for name, expected in (("suite", result.suite), ("config", result.config)):
                    if row.get(name) != expected:
                        issues.append(Issue("error", f"{name} {row.get(name)!r} != directory {expected!r}"))
            for issue in issues:
                add(line_no, issue)
    if result.rows == 0:
        add(0, Issue("error", "file has no rows"))
    return result


def sample_records(path: Path, n: int = LIVE_SAMPLE_ROWS) -> list[BenchRecord]:
    """Decode the first ``n`` rows of a file (a deterministic sample).

    Args:
        path: ``<split>.jsonl`` file.
        n: Number of rows.

    Returns:
        Wire-ready records.
    """
    records: list[BenchRecord] = []
    for row in read_rows(path):
        if len(records) >= n:
            break
        records.append(decode_row(row))
    return records


def live_check(config_key: str, records: Iterable[BenchRecord], client: SystemOneClient) -> LiveResult:
    """POST each record to a backend and check the answer has the record's question kind.

    Args:
        config_key: ``<suite>/<config>`` label for reporting.
        records: Decoded records.
        client: Client for the backend under test.

    Returns:
        Per-config pass counts and failure messages.
    """
    result = LiveResult(config_key=config_key)
    for record in records:
        result.sent += 1
        try:
            evaluation = client.evaluate(record.state, {LIVE_QUESTION_ID: record.question})
            dist = answer_distribution(evaluation.answers[LIVE_QUESTION_ID], record.kind, record.options)
        except (SystemOneError, AnswerParseError, httpx.HTTPError) as exc:
            result.failures.append(f"{record.id}: {type(exc).__name__}: {exc}")
            continue
        if not all(0.0 <= p <= 1.0 for p in dist.values()):
            result.failures.append(f"{record.id}: probabilities outside [0, 1]")
            continue
        result.passed += 1
    return result


def render_summary(results: Iterable[FileResult], live: Mapping[str, LiveResult] | None = None) -> str:
    """Render a plain-text per-file summary table.

    Args:
        results: Structural results.
        live: Live results keyed by ``<suite>/<config>``; the column is omitted when ``None``.

    Returns:
        The table followed by any issue messages.
    """
    header = ["suite/config", "split", "rows", "kinds", "errors", "warnings"]
    if live is not None:
        header.append("live")
    table: list[list[str]] = [header]
    messages: list[str] = []
    for r in results:
        key = f"{r.suite}/{r.config}"
        kinds = ",".join(f"{k}:{n}" for k, n in sorted(r.kinds.items()))
        line = [key, r.split, str(r.rows), kinds, str(r.errors), str(r.warnings)]
        if live is not None:
            lr = live.get(key)
            line.append(f"{lr.passed}/{lr.sent}" if lr and r.split == "test" else "-")
        table.append(line)
        messages += r.messages
    if live is not None:
        messages += [msg for lr in live.values() for msg in lr.failures]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip() for row in table]
    return "\n".join([*lines, *messages]) + "\n"
