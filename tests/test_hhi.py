"""Tests for concentration metrics."""

import math

import pytest

from covered import hhi


def test_hhi_single_source_is_one() -> None:
    assert hhi.hhi([42]) == pytest.approx(1.0)


def test_hhi_uniform_n_is_one_over_n() -> None:
    assert hhi.hhi([1, 1, 1, 1]) == pytest.approx(0.25)
    assert hhi.hhi([5, 5]) == pytest.approx(0.5)


def test_hhi_known_distribution() -> None:
    # shares 0.5, 0.3, 0.2 -> 0.25 + 0.09 + 0.04 = 0.38
    assert hhi.hhi([50, 30, 20]) == pytest.approx(0.38)


def test_hhi_ignores_zeros() -> None:
    assert hhi.hhi([50, 30, 20, 0, 0]) == pytest.approx(0.38)


def test_hhi_rejects_negative() -> None:
    with pytest.raises(ValueError, match="negative"):
        hhi.hhi([1, -2, 3])


def test_normalized_hhi_uniform_is_zero() -> None:
    assert hhi.normalized_hhi([1, 1, 1, 1]) == pytest.approx(0.0)


def test_normalized_hhi_single_is_one() -> None:
    assert hhi.normalized_hhi([7]) == pytest.approx(1.0)


def test_top_k_share() -> None:
    counts = {"a": 50, "b": 30, "c": 20}
    assert hhi.top_k_share(counts, 1) == pytest.approx(0.5)
    assert hhi.top_k_share(counts, 2) == pytest.approx(0.8)
    assert hhi.top_k_share(counts, 10) == pytest.approx(1.0)  # k exceeds N


def test_normalized_entropy_uniform_is_one() -> None:
    assert hhi.normalized_entropy([1, 1, 1, 1]) == pytest.approx(1.0)


def test_normalized_entropy_single_is_zero() -> None:
    assert hhi.normalized_entropy([9]) == pytest.approx(0.0)


def test_n_distinct_ignores_zeros() -> None:
    assert hhi.n_distinct([3, 0, 5, 0]) == 2


def test_empty_inputs() -> None:
    assert hhi.n_distinct([]) == 0
    assert hhi.top_k_share({}, 3) == 0.0
    assert math.isnan(hhi.hhi([]))


def test_concentration_metrics_bundle() -> None:
    counts = {chr(97 + i): 1 for i in range(30)}  # 30 equal sources
    m = hhi.concentration_metrics(counts)
    assert m["n_distinct"] == 30
    assert m["n_events"] == 30
    assert m["hhi"] == pytest.approx(1 / 30)
    assert m["hhi_normalized"] == pytest.approx(0.0)
    assert m["entropy_norm"] == pytest.approx(1.0)
    assert m["top10_share"] == pytest.approx(10 / 30)
    assert m["top25_share"] == pytest.approx(25 / 30)
