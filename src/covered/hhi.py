"""Concentration metrics for source counts.

The headline measure is the Herfindahl-Hirschman Index, but raw HHI is sensitive
to the long tail of one-off sources, so it is always reported alongside
normalized HHI, top-k share, and normalized entropy. The narrative rests on
these agreeing, not on raw HHI alone.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np

__all__ = [
    "concentration_metrics",
    "hhi",
    "n_distinct",
    "normalized_entropy",
    "normalized_hhi",
    "top_k_share",
]


def _positive(counts: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(counts), dtype=float)
    if arr.size and (arr < 0).any():
        raise ValueError("counts must be non-negative")
    return arr[arr > 0]


def hhi(counts: Iterable[float]) -> float:
    """Herfindahl-Hirschman Index: sum of squared market shares (0-1).

    Returns NaN for an empty market. Single source -> 1.0; ``n`` equal sources
    -> ``1/n``.
    """
    arr = _positive(counts)
    if arr.size == 0:
        return math.nan
    shares = arr / arr.sum()
    return float(np.square(shares).sum())


def normalized_hhi(counts: Iterable[float]) -> float:
    """HHI rescaled to [0, 1] independent of market size ``N``.

    ``(HHI - 1/N) / (1 - 1/N)``. A single source -> 1.0; ``N`` equal sources
    -> 0.0. Returns NaN for an empty market.
    """
    arr = _positive(counts)
    n = arr.size
    if n == 0:
        return math.nan
    if n == 1:
        return 1.0
    raw = hhi(arr)
    return float((raw - 1.0 / n) / (1.0 - 1.0 / n))


def top_k_share(counts: Mapping[str, float] | Iterable[float], k: int) -> float:
    """Share of total held by the ``k`` largest sources (0-1)."""
    values = counts.values() if isinstance(counts, Mapping) else counts
    arr = _positive(values)
    if arr.size == 0:
        return 0.0
    top = np.sort(arr)[::-1][:k]
    return float(top.sum() / arr.sum())


def normalized_entropy(counts: Iterable[float]) -> float:
    """Shannon entropy divided by ``log(N)`` (0-1).

    Uniform -> 1.0 (maximally diverse); single source -> 0.0. Returns NaN for an
    empty market.
    """
    arr = _positive(counts)
    n = arr.size
    if n == 0:
        return math.nan
    if n == 1:
        return 0.0
    shares = arr / arr.sum()
    entropy = float(-(shares * np.log(shares)).sum())
    return entropy / math.log(n)


def n_distinct(counts: Iterable[float]) -> int:
    """Number of sources with a positive count."""
    return int(_positive(counts).size)


def concentration_metrics(counts: Mapping[str, float]) -> dict[str, float]:
    """Bundle every concentration metric for one market (e.g. one year)."""
    values = list(counts.values())
    return {
        "hhi": hhi(values),
        "hhi_normalized": normalized_hhi(values),
        "top10_share": top_k_share(values, 10),
        "top25_share": top_k_share(values, 25),
        "entropy_norm": normalized_entropy(values),
        "n_distinct": float(n_distinct(values)),
        "n_events": float(sum(v for v in values if v > 0)),
    }
