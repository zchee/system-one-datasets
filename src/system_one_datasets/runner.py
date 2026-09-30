"""Sequential, resumable benchmark runner that writes one JSONL row per record."""

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, TypedDict

import httpx
import orjson
from tqdm import tqdm

from system_one_datasets.client import AnswerParseError, SystemOneClient, SystemOneError, answer_distribution
from system_one_datasets.metrics import top_option


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from system_one_datasets.schema import BenchRecord, Kind


logger = logging.getLogger(__name__)

QUESTION_ID = "q"
_FATAL_STATUSES = frozenset({httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN})


class ResultRow(TypedDict):
    """One line of a results JSONL file."""

    id: str
    config: str
    kind: Kind
    label: str
    soft_label: dict[str, float] | None
    predicted: dict[str, float] | None
    top: str | None
    latency_s: float | None
    usage: dict[str, int] | None
    model: str | None
    error: str | None


@dataclass(slots=True)
class RunSummary:
    """Counts from one ``run`` call."""

    skipped: int = 0
    succeeded: int = 0
    failed: int = 0


def iter_result_rows(path: Path) -> Iterator[ResultRow]:
    """Yield rows of a results JSONL file, skipping blank or unparsable lines (e.g. a torn final write).

    Args:
        path: Results file.

    Yields:
        Parsed rows in file order.
    """
    with path.open("rb") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError:
                logger.warning("%s:%d: skipping unparsable line", path, lineno)


def _ends_mid_line(path: Path) -> bool:
    size = path.stat().st_size
    if size == 0:
        return False
    with path.open("rb") as fh:
        fh.seek(size - 1)
        return fh.read(1) != b"\n"


def completed_ids(path: Path) -> set[str]:
    """Return ids that already have a successful row in ``path``.

    Rows with a non-null ``error`` do not count, so failed records are retried on the next run; the report keeps
    the last row per id.

    Args:
        path: Results file; a missing file yields an empty set.

    Returns:
        Ids to skip.
    """
    if not path.exists():
        return set()
    return {row["id"] for row in iter_result_rows(path) if row.get("error") is None}


def evaluate_record(client: SystemOneClient, record: BenchRecord) -> ResultRow:
    """Send one record to the backend and build its result row.

    Per-record failures (4xx other than auth, exhausted retries, malformed answers) become an ``error`` row.

    Args:
        client: Backend client.
        record: Benchmark record.

    Returns:
        The result row.

    Raises:
        SystemOneError: On 401/403, which would fail every remaining record.
    """
    row = ResultRow(
        id=record.id,
        config=record.config,
        kind=record.kind,
        label=record.label,
        soft_label=record.soft_label,
        predicted=None,
        top=None,
        latency_s=None,
        usage=None,
        model=None,
        error=None,
    )
    try:
        evaluation = client.evaluate(record.state, {QUESTION_ID: record.question})
        dist = answer_distribution(evaluation.answers[QUESTION_ID], record.kind, record.options)
    except SystemOneError as exc:
        if exc.status in _FATAL_STATUSES:
            raise
        row["error"] = str(exc)
    except (httpx.HTTPError, AnswerParseError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    else:
        row.update(
            predicted=dist,
            top=top_option(dist, record.kind),
            latency_s=evaluation.latency_s,
            usage=evaluation.usage,
            model=evaluation.model,
        )
    return row


def run(
    records: Iterable[BenchRecord], client: SystemOneClient, out_path: Path, *, progress: bool = True
) -> RunSummary:
    """Evaluate records sequentially, appending one JSONL row each and skipping ids already completed.

    Each row is flushed as soon as it is written, so an interrupted run resumes where it stopped.

    Args:
        records: Records to evaluate.
        client: Backend client.
        out_path: Results JSONL file (created, with parent directories, if missing).
        progress: Show a tqdm progress bar.

    Returns:
        Counts of skipped, succeeded, and failed records.
    """
    done = completed_ids(out_path)
    records = list(records)
    pending = [r for r in records if r.id not in done]
    summary = RunSummary(skipped=len(records) - len(pending))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("ab") as fh:
        if _ends_mid_line(out_path):
            fh.write(b"\n")  # Terminate a torn final write so the next row starts on its own line.
        for record in tqdm(pending, desc=f"{client.model} -> {out_path.name}", unit="req", disable=not progress):
            row = evaluate_record(client, record)
            fh.write(orjson.dumps(row) + b"\n")
            fh.flush()
            if row["error"] is None:
                summary.succeeded += 1
            else:
                summary.failed += 1
                logger.warning("%s: %s", record.id, row["error"])
    return summary
