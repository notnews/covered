"""Tests for validation metrics and error-adjusted HHI simulation.

These are unit tests: they exercise branches and check arithmetic on fixed
inputs. Whether either interval actually covers at the rate it claims is a
different question, and it is answered by the Monte Carlo studies in
``tests/test_metrics_coverage.py``. Three tests here were renamed to say what
they really check, because their original names claimed more than their bodies
could deliver.
"""

import math

import numpy as np
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


def test_bootstrap_ci_of_a_constant_sample_is_a_point() -> None:
    """Was ``test_bootstrap_ci_brackets_mean``, which it never did.

    Every resample of a constant vector is that constant, so both endpoints
    equal 0.4 by construction and this input cannot distinguish a correct
    percentile bootstrap from ``return (lo, lo)``. Kept for the degenerate case;
    coverage is checked in ``tests/test_metrics_coverage.py``.
    """
    lo, hi = metrics.bootstrap_ci([0.4] * 100, n_boot=200, seed=1)
    assert lo == pytest.approx(0.4, abs=1e-9)
    assert hi == pytest.approx(0.4, abs=1e-9)


def test_bootstrap_ci_widens_with_the_confidence_level() -> None:
    """One thing the constant-sample test above cannot see: ``alpha`` is used.

    Same sample, same resampling seed, so the two intervals are quantiles of the
    same bootstrap distribution and must nest strictly.
    """
    samples = np.random.default_rng(0).normal(0.4, 0.2, 200)
    wide = metrics.bootstrap_ci(samples, n_boot=999, alpha=0.05, seed=3)
    narrow = metrics.bootstrap_ci(samples, n_boot=999, alpha=0.20, seed=3)
    assert wide[0] < narrow[0] < narrow[1] < wide[1]


def test_simulate_hhi_skips_the_simulation_when_extraction_is_perfect() -> None:
    """Was ``test_simulate_hhi_no_error_is_deterministic``.

    ``recall=1.0, precision=1.0`` is the branch where ``kept = obs`` and
    ``add_factor = 0``, so nothing is simulated and all three quantiles collapse
    onto ``hhi(counts)``. That is a branch test, not evidence about the interval.
    """
    counts = {"a": 50, "b": 30, "c": 20}
    res = metrics.simulate_hhi_ci(counts, recall=1.0, precision=1.0, n_draws=50, seed=7)
    assert res["median"] == pytest.approx(0.38)
    assert res["lo"] == pytest.approx(0.38)
    assert res["hi"] == pytest.approx(0.38)


def test_simulate_hhi_with_error_returns_a_non_degenerate_interval() -> None:
    """Was ``test_simulate_hhi_with_error_brackets_and_orders``.

    ``lo <= median <= hi`` holds for any three sorted quantiles of any array, so
    the original assertion was true of an all-zero ``draws``. The strict
    inequality is the one that requires the simulation to have run and produced
    spread.
    """
    counts: dict[str, float] = {f"s{i}": 1 for i in range(20)}
    counts["dominant"] = 80
    res = metrics.simulate_hhi_ci(
        counts, recall=0.7, precision=0.9, n_draws=200, seed=3
    )
    assert res["lo"] < res["median"] < res["hi"]
    assert not math.isnan(res["median"])
