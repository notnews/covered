"""Plot the annual HHI series and coverage diagnostics.

Era boundaries (2002, 2014) are annotated on every figure and the partial final
year is shaded, so format/coverage shifts are not misread as real changes in
source concentration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.axes import Axes

matplotlib.use("Agg")  # headless: no display needed for file output
import matplotlib.pyplot as plt
import pandas as pd

__all__ = [
    "ERA_BOUNDARY_YEARS",
    "plot_clip_share",
    "plot_coverage",
    "plot_effective_voices",
    "plot_hhi_series",
    "plot_president_share",
    "plot_topk_bands",
]

ERA_BOUNDARY_YEARS = (2002, 2014)
PARTIAL_YEAR = 2025


def _annotate_eras(ax: Axes) -> None:
    for year in ERA_BOUNDARY_YEARS:
        ax.axvline(year, color="0.7", linestyle="--", linewidth=1, zorder=0)
    ax.axvspan(PARTIAL_YEAR - 0.5, PARTIAL_YEAR + 0.5, color="0.9", zorder=0)


def plot_hhi_series(
    annual: pd.DataFrame,
    out_path: Path,
    value_col: str = "hhi",
    title: str | None = None,
) -> Path:
    """Plot ``value_col`` over year, one line per (measure, variant)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    _annotate_eras(ax)
    for (measure, variant), grp in annual.groupby(["measure", "variant"]):
        g = grp.sort_values("year")
        ax.plot(g["year"], g[value_col], marker="o", label=f"{measure}:{variant}")
    ax.set_xlabel("Year")
    ax.set_ylabel(value_col)
    ax.set_title(title or f"CNN source concentration ({value_col}), 2000-2025")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_topk_bands(
    modes: pd.DataFrame,
    out_path: Path,
    variant: str = "external-clip",
    title: str | None = None,
) -> Path:
    """Stacked proportion of guest turns: top-10 / next-15 / rest, over year.

    ``modes`` is the ``hhi_speakers_modes.csv`` frame (columns ``year``,
    ``variant``, ``top10_share``, ``top25_share``). The bands sum to 1 and make
    the concentration of a single ``variant`` legible as the share of airtime the
    busiest sources capture.
    """
    g = modes[modes["variant"] == variant].sort_values("year")
    if g.empty:
        raise ValueError(f"no rows for variant {variant!r}")
    year = g["year"].to_numpy()
    top10 = g["top10_share"].to_numpy()
    next15 = (g["top25_share"] - g["top10_share"]).to_numpy()
    rest = (1.0 - g["top25_share"]).to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    _annotate_eras(ax)
    ax.stackplot(
        year,
        top10,
        next15,
        rest,
        labels=["Top 10 sources", "Next 15 (11-25)", "Everyone else"],
        colors=["#cc3311", "#ee9988", "#cfe0ee"],
    )
    ax.set_xlim(year.min(), year.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of guest turns")
    ax.set_title(title or f"Share of {variant} guest turns by source rank")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_effective_voices(modes: pd.DataFrame, out_path: Path) -> Path:
    """Effective number of voices (1/HHI) for live vs clip, over year.

    More legible than raw HHI: a flat line in the hundreds (live booking) against
    a line that collapses toward ~20 during the clip-concentration spike. Plotted
    on a log y-axis because the two regimes span two orders of magnitude.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    _annotate_eras(ax)
    styles = {
        "external-live": ("#0077bb", "Live (booked)"),
        "external-clip": ("#cc3311", "Clip (replayed)"),
    }
    for variant, (color, label) in styles.items():
        g = modes[modes["variant"] == variant].sort_values("year")
        if g.empty:
            continue
        ax.plot(g["year"], 1.0 / g["hhi"], marker="o", color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Effective number of voices (1/HHI)")
    ax.set_title("Effective number of external voices: live vs clip")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_clip_share(modes: pd.DataFrame, out_path: Path) -> Path:
    """Clip turns as a share of all guest turns (live + clip), over year.

    Documents the channel-mix shift: even at constant per-clip concentration, a
    rising clip share raises the aggregate weight of replayed elite voices.
    """
    live = modes[modes["variant"] == "external-live"][["year", "n_events"]]
    clip = modes[modes["variant"] == "external-clip"][["year", "n_events"]]
    merged = live.merge(clip, on="year", suffixes=("_live", "_clip")).sort_values(
        "year"
    )
    if merged.empty:
        raise ValueError("no overlapping live/clip years")
    share = merged["n_events_clip"] / (
        merged["n_events_clip"] + merged["n_events_live"]
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    _annotate_eras(ax)
    ax.fill_between(merged["year"], share, color="#cc3311", alpha=0.25)
    ax.plot(merged["year"], share, marker="o", color="#cc3311")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Clip share of guest turns")
    ax.set_title("Replayed clips as a share of all external guest turns")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _shade_administrations(ax: Axes, pres: pd.DataFrame) -> None:
    """Shade contiguous runs of years sharing a ``president_id`` and label them."""
    runs: list[tuple[str, int, int]] = []
    for _, r in pres.sort_values("year").iterrows():
        pid = "" if pd.isna(r["president_id"]) else str(r["president_id"])
        year = int(r["year"])
        if runs and runs[-1][0] == pid:
            runs[-1] = (pid, runs[-1][1], year)
        else:
            runs.append((pid, year, year))
    for i, (pid, y0, y1) in enumerate(runs):
        if not pid:
            continue
        ax.axvspan(y0 - 0.5, y1 + 0.5, color="0.93" if i % 2 else "0.85", zorder=0)
        ax.text(
            (y0 + y1) / 2,
            0.97,
            pid.split()[-1].title(),
            ha="center",
            va="top",
            fontsize=8,
            color="0.4",
            transform=ax.get_xaxis_transform(),
        )


def plot_president_share(
    pres: pd.DataFrame, out_path: Path, mode: str = "clip"
) -> Path:
    """Share of guest turns spoken by the sitting US president, over year.

    ``pres`` is the ``president_share_annual.csv`` frame (columns ``year``,
    ``mode``, ``president_id``, ``president_share``). Administrations are shaded
    and labelled so the spike maps onto who holds the office.
    """
    g = pres[pres["mode"] == mode].sort_values("year")
    if g.empty:
        raise ValueError(f"no rows for mode {mode!r}")
    fig, ax = plt.subplots(figsize=(9, 5))
    _shade_administrations(ax, g)
    ax.plot(g["year"], g["president_share"], marker="o", color="#aa3377")
    ax.set_ylim(0, max(0.05, g["president_share"].max() * 1.15))
    ax.set_xlabel("Year")
    ax.set_ylabel("President's share of guest turns")
    ax.set_title(f"Sitting president's share of external {mode} turns")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_coverage(coverage: pd.DataFrame, out_path: Path) -> Path:
    """Plot per-year segment counts (a coverage diagnostic).

    ``coverage`` has columns ``year`` and ``n_segments``.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    _annotate_eras(ax)
    g = coverage.sort_values("year")
    ax.bar(g["year"], g["n_segments"], color="#4477aa")
    ax.set_xlabel("Year")
    ax.set_ylabel("Segments")
    ax.set_title("Corpus coverage by year (diagnostic)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
