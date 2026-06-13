"""Tests for validation metrics and error-adjusted HHI simulation."""

import math

import pytest

from covered import metrics


def test_prf_basic() -> None:
    r = metrics.prf(tp=8, fp=2, fn=2)
    assert r["precision"] == pytest.approx(0.8)
    assert r["recall"] == pytest.approx(0.8)
    assert r["f1"] == pytest.approx(0.8)


def test_prf_zero_denominators() -> None:
    r = metrics.prf(tp=0, fp=0, fn=0)
    assert r["precision"] == 0.0
    assert r["recall"] == 0.0
    assert r["f1"] == 0.0


def test_bootstrap_ci_brackets_mean() -> None:
    lo, hi = metrics.bootstrap_ci([0.4] * 100, n_boot=200, seed=1)
    assert lo == pytest.approx(0.4, abs=1e-9)
    assert hi == pytest.approx(0.4, abs=1e-9)


def test_simulate_hhi_no_error_is_deterministic() -> None:
    counts = {"a": 50, "b": 30, "c": 20}
    res = metrics.simulate_hhi_ci(counts, recall=1.0, precision=1.0, n_draws=50, seed=7)
    assert res["median"] == pytest.approx(0.38)
    assert res["lo"] == pytest.approx(0.38)
    assert res["hi"] == pytest.approx(0.38)


def test_simulate_hhi_with_error_brackets_and_orders() -> None:
    counts = {f"s{i}": 1 for i in range(20)}
    counts["dominant"] = 80
    res = metrics.simulate_hhi_ci(counts, recall=0.7, precision=0.9, n_draws=200, seed=3)
    assert res["lo"] <= res["median"] <= res["hi"]
    assert not math.isnan(res["median"])
