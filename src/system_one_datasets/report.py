"""Markdown report over one or more results JSONL files."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from system_one_datasets.metrics import (
    ECE_BINS,
    accuracy,
    brier_score,
    expected_calibration_error,
    mean_latency,
    p95_latency,
    ranked_probability_score,
    total_variation_distance,
)
from system_one_datasets.runner import ResultRow, iter_result_rows


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


OVERALL = "overall"
_COLUMNS = ("source", "model", "n", "errors", "accuracy", "ECE", "Brier", "RPS", "TVD", "mean_lat_s", "p95_lat_s")


@dataclass(frozen=True, slots=True)
class Summary:
    """Metrics for one (results file, config) group; ``None`` means not applicable or no data."""

    source: str
    model: str
    n: int
    errors: int
    accuracy: float | None
    ece: float | None
    brier: float | None
    rps: float | None
    tvd: float | None
    mean_latency_s: float | None
    p95_latency_s: float | None


def latest_rows(path: Path) -> list[ResultRow]:
    """Read a results file keeping only the last row per id, so a retried record replaces its failed attempt.

    Args:
        path: Results JSONL file.

    Returns:
        One row per id, in first-seen order.
    """
    rows: dict[str, ResultRow] = {}
    for row in iter_result_rows(path):
        rows[row["id"]] = row
    return list(rows.values())


def summarize(source: str, rows: Sequence[ResultRow]) -> Summary:
    """Compute every metric for a group of rows; error rows are counted but excluded from metrics.

    Args:
        source: Display name of the results file.
        rows: Result rows of one config (or all configs, for the overall group).

    Returns:
        The group summary.
    """
    ok = [r for r in rows if r["error"] is None and r["predicted"] is not None and r["top"] is not None]
    dists = [r["predicted"] or {} for r in ok]
    labels = [r["label"] for r in ok]
    tops = [r["top"] or "" for r in ok]
    score_rows = [r for r in ok if r["kind"] == "score"]
    latencies = [r["latency_s"] for r in ok if r["latency_s"] is not None]
    models = sorted({r["model"] for r in ok if r["model"]})
    return Summary(
        source=source,
        model=",".join(models) or "-",
        n=len(ok),
        errors=len(rows) - len(ok),
        accuracy=accuracy(tops, labels),
        ece=expected_calibration_error(
            [d[t] for d, t in zip(dists, tops, strict=True)], [t == y for t, y in zip(tops, labels, strict=True)]
        ),
        brier=brier_score(dists, labels),
        rps=ranked_probability_score([r["predicted"] or {} for r in score_rows], [r["label"] for r in score_rows]),
        tvd=total_variation_distance(dists, [r["soft_label"] for r in ok]),
        mean_latency_s=mean_latency(latencies),
        p95_latency_s=p95_latency(latencies),
    )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _table(summaries: Sequence[Summary]) -> list[str]:
    lines = ["| " + " | ".join(_COLUMNS) + " |", "|" + "|".join(["---"] * 2 + ["---:"] * (len(_COLUMNS) - 2)) + "|"]
    lines.extend(
        "| "
        + " | ".join([
            s.source,
            s.model,
            str(s.n),
            str(s.errors),
            _fmt(s.accuracy),
            _fmt(s.ece),
            _fmt(s.brier),
            _fmt(s.rps),
            _fmt(s.tvd),
            _fmt(s.mean_latency_s),
            _fmt(s.p95_latency_s),
        ])
        + " |"
        for s in summaries
    )
    return lines


def render_report(paths: Sequence[Path]) -> str:
    """Render Markdown tables per config and overall, one table row per results file.

    Args:
        paths: Results JSONL files (typically one per backend).

    Returns:
        The Markdown report.
    """
    by_source = {path.stem: latest_rows(path) for path in paths}
    configs = sorted({r["config"] for rows in by_source.values() for r in rows})
    lines = [
        "# System One benchmark report",
        "",
        "Metrics exclude error rows. Brier is the multiclass sum form (0..2). RPS covers score rows only; "
        f"TVD covers rows with soft labels only. ECE uses {ECE_BINS} equal-width bins on top-option probability.",
    ]
    for group in [*configs, OVERALL]:
        lines += ["", f"## {group}", ""]
        lines += _table([
            summarize(source, rows if group == OVERALL else [r for r in rows if r["config"] == group])
            for source, rows in by_source.items()
        ])
    return "\n".join(lines) + "\n"
