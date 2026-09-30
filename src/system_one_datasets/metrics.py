"""Pure scoring functions over predicted distributions and gold labels.

Every distribution is a ``dict[option, probability]`` whose keys are the question's full option set. Functions
return ``None`` when there is nothing to score.
"""

from collections.abc import Mapping, Sequence
import math
import statistics

from system_one_datasets.schema import NOUL_OPTIONS, Kind


type Distribution = Mapping[str, float]

ECE_BINS = 15


def top_option(dist: Distribution, kind: Kind) -> str:
    """Return the argmax option of ``dist``.

    The prediction is always derived from the distribution (never from a backend's ``choice`` field) so every
    backend is scored identically. Noul predicts ``"1"`` when P(yes) >= 0.5, the same threshold jev-bench used to
    derive its hard labels. Other ties go to the earliest option in ``dist`` order.

    Args:
        dist: Probability per option.
        kind: Question type.

    Returns:
        The predicted option key.

    Raises:
        ValueError: ``dist`` is empty.
    """
    if not dist:
        msg = "empty distribution"
        raise ValueError(msg)
    if kind == "noul":
        no, yes = NOUL_OPTIONS
        return yes if dist.get(yes, 0.0) >= 0.5 else no
    return max(dist, key=lambda option: dist[option])


def accuracy(predicted: Sequence[str], labels: Sequence[str]) -> float | None:
    """Fraction of predictions equal to their label.

    Args:
        predicted: Predicted option per row.
        labels: Gold option per row.

    Returns:
        Accuracy in [0, 1].
    """
    if not labels:
        return None
    return sum(p == y for p, y in zip(predicted, labels, strict=True)) / len(labels)


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = ECE_BINS
) -> float | None:
    """Expected calibration error with equal-width confidence bins.

    Row ``i`` falls in bin ``min(floor(conf_i * n_bins), n_bins - 1)``; the result is
    ``sum_b (n_b / N) * |acc_b - mean_conf_b|``.

    Args:
        confidences: Probability of the predicted (top) option per row.
        correct: Whether the prediction matched the label per row.
        n_bins: Number of equal-width bins on [0, 1].

    Returns:
        ECE in [0, 1].
    """
    n = len(confidences)
    if n == 0:
        return None
    counts = [0] * n_bins
    conf_sums = [0.0] * n_bins
    hits = [0] * n_bins
    for conf, ok in zip(confidences, correct, strict=True):
        b = min(int(conf * n_bins), n_bins - 1)
        counts[b] += 1
        conf_sums[b] += conf
        hits[b] += ok
    return sum(abs(hits[b] - conf_sums[b]) for b in range(n_bins) if counts[b]) / n


def brier_score(dists: Sequence[Distribution], labels: Sequence[str]) -> float | None:
    """Multiclass Brier score: mean over rows of ``sum_k (p_k - y_k)^2`` with one-hot ``y``.

    This is the sum form (range 0..2), not divided by the number of classes.

    Args:
        dists: Predicted distribution per row.
        labels: Gold option per row; must be a key of the row's distribution.

    Returns:
        Mean Brier score.

    Raises:
        ValueError: A label is not an option of its distribution.
    """
    if not labels:
        return None
    total = 0.0
    for dist, label in zip(dists, labels, strict=True):
        if label not in dist:
            msg = f"label {label!r} not in distribution"
            raise ValueError(msg)
        total += sum((p - (option == label)) ** 2 for option, p in dist.items())
    return total / len(labels)


def ranked_probability_score(dists: Sequence[Distribution], labels: Sequence[str]) -> float | None:
    """Ranked probability score for ordered score levels.

    Levels are the distribution keys ``"0".."K-1"`` ordered numerically. Per row,
    ``RPS = sum_{k<K-1} (CDF_pred_k - CDF_true_k)^2 / (K - 1)``; the result is the mean over rows (range 0..1).

    Args:
        dists: Predicted distribution over level indices per row.
        labels: Gold level index per row.

    Returns:
        Mean RPS.

    Raises:
        ValueError: A distribution has fewer than two levels or the label is not a level.
    """
    if not labels:
        return None
    total = 0.0
    for dist, label in zip(dists, labels, strict=True):
        levels = sorted(dist, key=int)
        if len(levels) < 2 or label not in dist:
            msg = "RPS needs >= 2 levels and a label among them"
            raise ValueError(msg)
        gold = int(label)
        cdf_pred = 0.0
        acc = 0.0
        for level in levels[:-1]:
            cdf_pred += dist[level]
            acc += (cdf_pred - (int(level) >= gold)) ** 2
        total += acc / (len(levels) - 1)
    return total / len(labels)


def total_variation_distance(dists: Sequence[Distribution], soft_labels: Sequence[Distribution | None]) -> float | None:
    """Mean total variation distance ``0.5 * sum_k |p_k - q_k|`` against annotator soft labels.

    Rows whose soft label is ``None`` are skipped; keys missing from either side count as 0.

    Args:
        dists: Predicted distribution per row.
        soft_labels: Annotator distribution per row, or ``None``.

    Returns:
        Mean TVD over rows that have a soft label, or ``None`` if none do.
    """
    values = [
        0.5 * sum(abs(dist.get(k, 0.0) - soft.get(k, 0.0)) for k in dist.keys() | soft.keys())
        for dist, soft in zip(dists, soft_labels, strict=True)
        if soft is not None
    ]
    return statistics.fmean(values) if values else None


def mean_latency(latencies: Sequence[float]) -> float | None:
    """Arithmetic mean latency in seconds.

    Args:
        latencies: Per-request latency in seconds.

    Returns:
        Mean latency.
    """
    return statistics.fmean(latencies) if latencies else None


def p95_latency(latencies: Sequence[float]) -> float | None:
    """95th-percentile latency by the nearest-rank method (``sorted[ceil(0.95 n) - 1]``).

    Args:
        latencies: Per-request latency in seconds.

    Returns:
        p95 latency; defined for any non-empty input, including a single sample.
    """
    if not latencies:
        return None
    ordered = sorted(latencies)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]
