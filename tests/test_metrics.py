"""Metric tests against hand-computed values."""

import pytest

from system_one_datasets.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    mean_latency,
    p95_latency,
    ranked_probability_score,
    top_option,
    total_variation_distance,
)


def test_top_option_noul_threshold_matches_dataset_labels() -> None:
    """Noul predicts "1" at exactly 0.5, matching how jev-bench derived hard labels from soft ones."""
    assert top_option({"0": 0.5, "1": 0.5}, "noul") == "1"
    assert top_option({"0": 0.51, "1": 0.49}, "noul") == "0"


def test_top_option_argmax_and_tie_break() -> None:
    """Choice/score take the argmax; ties go to the earliest option."""
    assert top_option({"a": 0.2, "b": 0.5, "c": 0.3}, "choice") == "b"
    assert top_option({"0": 0.4, "1": 0.4, "2": 0.2}, "score") == "0"
    with pytest.raises(ValueError, match="empty"):
        top_option({}, "choice")


def test_accuracy() -> None:
    """Two of three correct."""
    assert accuracy(["a", "b", "a"], ["a", "a", "a"]) == pytest.approx(2 / 3)
    assert accuracy([], []) is None


def test_expected_calibration_error() -> None:
    """Hand-computed.

    conf 0.9 -> bin 13 (13.5 floored): 2 rows, acc 0.5, conf 0.9 -> 0.5 * 0.4 = 0.20.
    conf 0.3 -> bin 4 (4.5 floored): 2 rows, acc 0.0, conf 0.3 -> 0.5 * 0.3 = 0.15.
    ECE = 0.35.
    """
    assert expected_calibration_error([0.9, 0.9, 0.3, 0.3], [True, False, False, False]) == pytest.approx(0.35)


def test_expected_calibration_error_edges() -> None:
    """A confidence of 1.0 lands in the last bin; a perfectly calibrated bin gives 0."""
    assert expected_calibration_error([1.0], [True]) == pytest.approx(0.0)
    assert expected_calibration_error([1.0, 1.0], [True, False]) == pytest.approx(0.5)
    assert expected_calibration_error([], []) is None


def test_brier_score() -> None:
    """Row 1: 0.3^2 + 0.2^2 + 0.1^2 = 0.14. Row 2: 0.5^2 + 0.5^2 = 0.50. Mean 0.32."""
    dists = [{"a": 0.7, "b": 0.2, "c": 0.1}, {"a": 0.5, "b": 0.5}]
    assert brier_score(dists, ["a", "b"]) == pytest.approx(0.32)
    with pytest.raises(ValueError, match="not in distribution"):
        brier_score([{"a": 1.0}], ["z"])


def test_ranked_probability_score() -> None:
    """Row 1 (gold 2): CDF_pred [0.2, 0.7] vs [0, 0] -> (0.04 + 0.49) / 2 = 0.265. Row 2 (gold 0, exact): 0.

    Mean 0.1325.
    """
    dists = [{"0": 0.2, "1": 0.5, "2": 0.3}, {"0": 1.0, "1": 0.0, "2": 0.0}]
    assert ranked_probability_score(dists, ["2", "0"]) == pytest.approx(0.1325)


def test_ranked_probability_score_orders_levels_numerically() -> None:
    """Keys "10" and "2" are ordered as numbers, not strings: CDF over 0,1,...,10."""
    dist = {str(i): 0.0 for i in range(11)}
    dist["10"] = 1.0
    assert ranked_probability_score([dist], ["10"]) == pytest.approx(0.0)
    assert ranked_probability_score([dist], ["0"]) == pytest.approx(1.0)


def test_total_variation_distance_skips_rows_without_soft_labels() -> None:
    """Row 1: 0.5 * (0.2 + 0.2) = 0.2. Row 2 skipped. Row 3: 0. Mean 0.1."""
    dists = [{"0": 0.7, "1": 0.3}, {"0": 0.1, "1": 0.9}, {"a": 0.6, "b": 0.4}]
    soft = [{"0": 0.5, "1": 0.5}, None, {"a": 0.6, "b": 0.4}]
    assert total_variation_distance(dists, soft) == pytest.approx(0.1)
    assert total_variation_distance(dists[:1], [None]) is None


def test_latency_stats() -> None:
    """Nearest-rank p95: n=20 -> rank 19; n=5 -> rank 5; n=1 -> the only sample."""
    assert p95_latency([float(i) for i in range(1, 21)]) == pytest.approx(19.0)
    assert p95_latency([5.0, 1.0, 3.0, 2.0, 4.0]) == pytest.approx(5.0)
    assert p95_latency([7.0]) == pytest.approx(7.0)
    assert p95_latency([]) is None
    assert mean_latency([1.0, 2.0, 6.0]) == pytest.approx(3.0)
    assert mean_latency([]) is None
