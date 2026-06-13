"""Validation metrics and error-adjusted HHI.

Extraction is imperfect: missed sources (recall < 1) drop mostly tail/one-off
voices and bias HHI *upward* (toward apparent concentration); false positives
(precision < 1) add phantom sources and bias it *downward*. We estimate
per-stratum precision/recall from the LLM-validation tier and propagate that
uncertainty into the HHI via Monte-Carlo simulation, reporting a corrected
series with credible intervals next to the raw one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from covered.hhi import hhi

__all__ = ["bootstrap_ci", "prf", "simulate_hhi_ci"]


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall, F1 from confusion counts (zero-safe)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def bootstrap_ci(
    samples: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``samples``."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def simulate_hhi_ci(
    counts: Mapping[str, float],
    recall: float,
    precision: float,
    n_draws: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Monte-Carlo error-adjusted HHI given estimated ``recall``/``precision``.

    Per draw and per observed source: remove false positives by keeping each
    observed event with probability ``precision``; then add back missed true
    events (~Poisson with mean ``kept * (1-recall)/recall``) to the same source.
    Missed events are conservatively attributed to already-observed sources, so
    the correction is a lower bound on tail diversity (logged as a caveat).
    """
    rng = np.random.default_rng(seed)
    sources = list(counts)
    obs = np.array([counts[s] for s in sources], dtype=float)
    draws = np.empty(n_draws, dtype=float)
    add_factor = (1.0 - recall) / recall if recall > 0 else 0.0

    for d in range(n_draws):
        kept = rng.binomial(obs.astype(int), precision) if precision < 1 else obs
        added = rng.poisson(kept * add_factor) if add_factor > 0 else np.zeros_like(kept)
        new = np.asarray(kept, dtype=float) + np.asarray(added, dtype=float)
        draws[d] = hhi(new)

    return {
        "median": float(np.nanmedian(draws)),
        "lo": float(np.nanquantile(draws, alpha / 2)),
        "hi": float(np.nanquantile(draws, 1 - alpha / 2)),
    }
