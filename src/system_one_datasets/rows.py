"""On-disk row format of ``data/<suite>/<config>/<split>.jsonl`` and its decoder.

``state``, ``question``, and ``soft_label`` are stored as JSON-encoded strings, the convention
``Praveenrajus/jev-bench`` uses. A Hugging Face ``datasets`` JSON loader infers one Arrow struct per column
across all rows; since ``question.criteria`` has different keys in every agent-action row, nested objects
would be unioned into one sparse struct and read back with null criteria for options that are not the
row's own. String columns give every config the same flat schema. ``soft_label`` is always a string: the
JSON text ``"null"`` when the row has no annotator distribution.
"""

from typing import TYPE_CHECKING

import orjson

from system_one_datasets.data import question_options
from system_one_datasets.schema import KINDS, BenchRecord


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from system_one_datasets.schema import JSONValue


# Field order of one row; ``encode_row`` writes keys in this order.
CANONICAL_FIELDS: tuple[str, ...] = (
    "id",
    "suite",
    "config",
    "kind",
    "state",
    "question",
    "options",
    "label",
    "soft_label",
    "source",
    "source_revision",
    "upstream",
    "license",
)
# Fields holding JSON-encoded text.
JSON_TEXT_FIELDS: tuple[str, ...] = ("state", "question", "soft_label")
# ``soft_label`` of a row without an annotator distribution.
NO_SOFT_LABEL = "null"


def dumps(value: object) -> str:
    """Compact JSON text (orjson; key order preserved, non-ASCII kept as UTF-8).

    Args:
        value: JSON-serializable value.

    Returns:
        The JSON text.
    """
    return orjson.dumps(value).decode()


def encode_row(
    record: BenchRecord, suite: str, source: str, revision: str, upstream: str, license_id: str
) -> dict[str, object]:
    """Build one on-disk row with keys in ``CANONICAL_FIELDS`` order.

    Args:
        record: Parsed record.
        suite: Suite directory name.
        source: Hugging Face dataset id the row was loaded from.
        revision: Commit sha of ``source``.
        upstream: Original dataset the row's content comes from (``source`` is a derivative of it).
        license_id: License of the row's content.

    Returns:
        The row.
    """
    return {
        "id": record.id,
        "suite": suite,
        "config": record.config,
        "kind": record.kind,
        "state": dumps(record.state),
        "question": dumps(record.question),
        "options": list(record.options),
        "label": record.label,
        "soft_label": dumps(record.soft_label),
        "source": source,
        "source_revision": revision,
        "upstream": upstream,
        "license": license_id,
    }


def decode_row(row: Mapping[str, object]) -> BenchRecord:
    """Decode an on-disk row into a wire-ready ``BenchRecord``.

    Args:
        row: One decoded JSONL line (or one ``datasets`` row).

    Returns:
        The record; ``state`` and ``question`` can be sent to ``POST /v1/systemone`` unchanged.

    Raises:
        TypeError: A field has the wrong JSON type.
        ValueError: A field is malformed, or ``options`` disagrees with the question.
    """
    texts: dict[str, str] = {}
    for name in JSON_TEXT_FIELDS:
        value = row.get(name)
        if not isinstance(value, str):
            msg = f"{name} must be a JSON-encoded string"
            raise TypeError(msg)
        texts[name] = value
    state: JSONValue = orjson.loads(texts["state"])
    question = orjson.loads(texts["question"])
    soft = orjson.loads(texts["soft_label"])
    kind = next((k for k in KINDS if k == row.get("kind")), None)
    if kind is None:
        msg = f"unknown kind {row.get('kind')!r}"
        raise ValueError(msg)
    if not isinstance(question, dict):
        msg = "question must encode an object"
        raise TypeError(msg)
    options = row.get("options")
    if not isinstance(options, list) or options != question_options(kind, question):
        msg = "options do not match the question"
        raise ValueError(msg)
    if soft is not None and (
        not isinstance(soft, dict)
        or not all(isinstance(p, int | float) and not isinstance(p, bool) for p in soft.values())
    ):
        msg = "soft_label must encode an object of numbers or null"
        raise TypeError(msg)
    return BenchRecord(
        id=str(row.get("id")),
        config=str(row.get("config")),
        kind=kind,
        state=state,
        question=question,
        options=[str(o) for o in options],
        label=str(row.get("label")),
        soft_label={str(k): float(v) for k, v in soft.items()} if soft is not None else None,
    )


def read_rows(path: Path) -> Iterator[dict[str, object]]:
    """Yield the raw rows of a JSONL file, skipping blank lines.

    Args:
        path: ``<split>.jsonl`` file.

    Yields:
        Decoded JSON objects (JSON-text fields still encoded).

    Raises:
        TypeError: A line is not a JSON object.
    """
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            row = orjson.loads(line)
            if not isinstance(row, dict):
                msg = f"{path}: line is not a JSON object"
                raise TypeError(msg)
            yield row


def load_records(path: Path) -> list[BenchRecord]:
    """Load and decode every row of a ``data/<suite>/<config>/<split>.jsonl`` file.

    Args:
        path: File to load.

    Returns:
        Wire-ready records in file order.
    """
    return [decode_row(row) for row in read_rows(path)]
