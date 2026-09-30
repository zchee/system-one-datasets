"""Runner resumability and report tests using a mock backend."""

from typing import TYPE_CHECKING

import httpx
import orjson
import pytest

from system_one_datasets.client import SystemOneClient, SystemOneError
from system_one_datasets.report import latest_rows, render_report
from system_one_datasets.runner import QUESTION_ID, completed_ids, run
from system_one_datasets.schema import BenchRecord, JSONValue


if TYPE_CHECKING:
    from pathlib import Path


QUESTION: dict[str, JSONValue] = {"type": "noul", "instructions": "Is this comment toxic?"}


def make_records(n: int) -> list[BenchRecord]:
    """Build ``n`` noul records alternating labels "1"/"0"."""
    return [
        BenchRecord(
            id=f"civil_comments/test/{i}",
            config="civil_comments",
            kind="noul",
            state=f"comment {i}",
            question=QUESTION.copy(),
            options=["0", "1"],
            label="1" if i % 2 == 0 else "0",
            soft_label={"0": 0.25, "1": 0.75},
        )
        for i in range(n)
    ]


def mock_client(seen: list[dict[str, object]], status: int = 200) -> SystemOneClient:
    """Client whose backend answers P(yes)=0.9 and records each decoded request body."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(orjson.loads(request.content))
        if status != 200:
            return httpx.Response(status, text="denied")
        body = {
            "model": "jev-1.13.0",
            "answers": {QUESTION_ID: {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        return httpx.Response(200, content=orjson.dumps(body))

    return SystemOneClient("https://example.test", "jev-1.13.0", transport=httpx.MockTransport(handler))


def seed_row(record: BenchRecord, error: str | None) -> bytes:
    """A results line as an earlier run would have written it."""
    row = {
        "id": record.id,
        "config": record.config,
        "kind": record.kind,
        "label": record.label,
        "soft_label": record.soft_label,
        "predicted": None if error else {"0": 0.1, "1": 0.9},
        "top": None if error else "1",
        "latency_s": None if error else 0.5,
        "usage": None if error else {"input_tokens": 10, "output_tokens": 2},
        "model": None if error else "jev-1.13.0",
        "error": error,
    }
    return orjson.dumps(row) + b"\n"


def test_run_writes_rows_with_expected_fields(tmp_path: Path) -> None:
    """A fresh run sends state/question verbatim and writes one complete row per record."""
    out = tmp_path / "results" / "jev.jsonl"
    seen: list[dict[str, object]] = []
    records = make_records(2)
    with mock_client(seen) as client:
        summary = run(records, client, out, progress=False)

    assert (summary.succeeded, summary.failed, summary.skipped) == (2, 0, 0)
    assert seen[0] == {"state": "comment 0", "model": "jev-1.13.0", "questions": {QUESTION_ID: QUESTION}}
    rows = [orjson.loads(line) for line in out.read_bytes().splitlines()]
    assert [r["id"] for r in rows] == [r.id for r in records]
    first = rows[0]
    assert first["predicted"] == {"0": pytest.approx(0.1), "1": 0.9}
    assert first["top"] == "1"
    assert first["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert first["model"] == "jev-1.13.0"
    assert first["error"] is None
    assert first["latency_s"] >= 0
    assert set(first) == {
        "id",
        "config",
        "kind",
        "label",
        "soft_label",
        "predicted",
        "top",
        "latency_s",
        "usage",
        "model",
        "error",
    }


def test_run_resumes_and_retries_failed_ids(tmp_path: Path) -> None:
    """Completed ids are skipped; an id whose only row is an error is retried; a torn last line is ignored."""
    out = tmp_path / "jev.jsonl"
    records = make_records(4)
    out.write_bytes(
        seed_row(records[0], None) + seed_row(records[1], None) + seed_row(records[2], "HTTP 503: x") + b'{"id": "torn'
    )
    assert completed_ids(out) == {records[0].id, records[1].id}

    seen: list[dict[str, object]] = []
    with mock_client(seen) as client:
        summary = run(records, client, out, progress=False)

    assert [body["state"] for body in seen] == ["comment 2", "comment 3"]
    assert (summary.succeeded, summary.failed, summary.skipped) == (2, 0, 2)
    latest = latest_rows(out)
    assert [r["id"] for r in latest] == [r.id for r in records]
    assert all(r["error"] is None for r in latest)

    with mock_client(seen) as client:
        again = run(records, client, out, progress=False)
    assert (again.succeeded, again.skipped) == (0, 4)
    assert len(seen) == 2


def test_auth_failure_aborts_the_run(tmp_path: Path) -> None:
    """401 is fatal: it raises instead of writing an error row for every record."""
    out = tmp_path / "jev.jsonl"
    with mock_client([], status=401) as client, pytest.raises(SystemOneError) as info:
        run(make_records(3), client, out, progress=False)
    assert info.value.status == 401
    assert out.read_bytes() == b""


def test_non_fatal_error_is_recorded(tmp_path: Path) -> None:
    """422 becomes an error row and the run continues."""
    out = tmp_path / "jev.jsonl"
    with mock_client([], status=422) as client:
        summary = run(make_records(2), client, out, progress=False)
    assert summary.failed == 2
    rows = [orjson.loads(line) for line in out.read_bytes().splitlines()]
    assert all(r["error"] == "HTTP 422: denied" and r["predicted"] is None for r in rows)


def test_report_tables(tmp_path: Path) -> None:
    """Report metrics on 4 noul rows, all predicting P(yes)=0.9 against labels 1,0,1,0 and soft label 0.75.

    accuracy 0.5; Brier mean of 0.02 and 1.62 = 0.82; ECE |0.5 - 0.9| = 0.4; TVD 0.15; RPS n/a for noul.
    """
    out = tmp_path / "jev.jsonl"
    with mock_client([]) as client:
        run(make_records(4), client, out, progress=False)
    with out.open("ab") as fh:
        fh.write(seed_row(make_records(5)[4], "HTTP 422: bad"))

    text = render_report([out])
    assert "## civil_comments" in text
    assert "## overall" in text
    line = next(ln for ln in text.splitlines() if ln.startswith("| jev |"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[:9] == ["jev", "jev-1.13.0", "4", "1", "0.5000", "0.4000", "0.8200", "-", "0.1500"]
    assert all(float(c) >= 0 for c in cells[9:])
