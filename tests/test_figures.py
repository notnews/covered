"""Smoke tests for plotting (Agg backend, writes to a temp dir)."""

from pathlib import Path

import pandas as pd
import pytest

from covered import figures


def test_plot_hhi_series_writes_file(tmp_path: Path) -> None:
    annual = pd.DataFrame(
        {
            "year": [2001, 2002, 2003, 2001, 2002, 2003],
            "measure": ["speakers"] * 3 + ["attributions"] * 3,
            "variant": ["external"] * 3 + ["person"] * 3,
            "hhi": [0.1, 0.12, 0.09, 0.05, 0.06, 0.04],
        }
    )
    out = figures.plot_hhi_series(annual, tmp_path / "hhi.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_coverage_writes_file(tmp_path: Path) -> None:
    coverage = pd.DataFrame({"year": [2000, 2001, 2002], "n_segments": [7000, 21000, 35000]})
    out = figures.plot_coverage(coverage, tmp_path / "coverage.png")
    assert out.exists() and out.stat().st_size > 0


def _modes_fixture() -> pd.DataFrame:
    """Minimal two-variant frame mirroring hhi_speakers_modes.csv columns."""
    return pd.DataFrame(
        {
            "year": [2016, 2017, 2018, 2016, 2017, 2018],
            "variant": ["external-clip"] * 3 + ["external-live"] * 3,
            "hhi": [0.021, 0.043, 0.044, 0.0017, 0.0023, 0.0020],
            "top10_share": [0.29, 0.33, 0.31, 0.083, 0.102, 0.095],
            "top25_share": [0.35, 0.39, 0.38, 0.152, 0.154, 0.144],
            "n_events": [109131, 104904, 103274, 177143, 147173, 139847],
        }
    )


def test_plot_topk_bands_writes_file(tmp_path: Path) -> None:
    out = figures.plot_topk_bands(_modes_fixture(), tmp_path / "bands.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_topk_bands_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="no rows for variant"):
        figures.plot_topk_bands(_modes_fixture(), Path("unused.png"), variant="external-nope")


def test_plot_effective_voices_writes_file(tmp_path: Path) -> None:
    out = figures.plot_effective_voices(_modes_fixture(), tmp_path / "voices.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_clip_share_writes_file(tmp_path: Path) -> None:
    out = figures.plot_clip_share(_modes_fixture(), tmp_path / "clip_share.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_president_share_writes_file(tmp_path: Path) -> None:
    pres = pd.DataFrame(
        {
            "year": [2016, 2017, 2018, 2019],
            "mode": ["clip"] * 4,
            "president_id": ["barack obama", "donald trump", "donald trump", "donald trump"],
            "president_share": [0.04, 0.11, 0.10, 0.09],
        }
    )
    out = figures.plot_president_share(pres, tmp_path / "pres.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_president_share_unknown_mode_raises() -> None:
    pres = pd.DataFrame(
        {"year": [2017], "mode": ["clip"], "president_id": ["x"], "president_share": [0.1]}
    )
    with pytest.raises(ValueError, match="no rows for mode"):
        figures.plot_president_share(pres, Path("unused.png"), mode="live")
