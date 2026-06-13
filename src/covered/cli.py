"""Command-line entry points for the covered pipeline.

Each stage reads from one data/ directory and writes the next, so stages are
independently re-runnable. ``pilot`` runs the whole chain on one date window and
asserts face validity (the sitting president must dominate cited sources).
"""

from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd
import typer

from covered import acquire, figures, pipeline
from covered.config import (
    FIGURES,
    INTERIM,
    PROCESSED,
    SPACY_MODEL_FULL,
    TABLES,
)

app = typer.Typer(add_completion=False, help="HHI of who gets quoted on CNN, 2000-2025.")

# Sitting US president by year — face-validity anchor for cited sources.
PRESIDENT_BY_YEAR = {
    **dict.fromkeys(range(2001, 2009), "george w. bush"),
    **dict.fromkeys(range(2009, 2017), "barack obama"),
    **dict.fromkeys(range(2017, 2021), "donald trump"),
    **dict.fromkeys(range(2021, 2025), "joe biden"),
    2025: "donald trump",
}


def _write(df: pd.DataFrame, path) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    typer.echo(f"wrote {len(df):,} rows -> {path}")


def _annual_all(turns: pd.DataFrame, atts: pd.DataFrame) -> pd.DataFrame:
    """Compute every headline + robustness annual series."""
    frames = [
        pipeline.annual_hhi_speakers(turns, variant="external"),
        pipeline.annual_hhi_speakers(turns, variant="all"),
        pipeline.annual_hhi_attributions(atts, entity_type="PERSON"),
        pipeline.annual_hhi_attributions(atts, entity_type="ORG"),
    ]
    return pd.concat(frames, ignore_index=True)


@app.command()
def acquire_data() -> None:
    """Download the 8 Dataverse CSVs (needs DATAVERSE_API_TOKEN)."""
    path = acquire.download_all()
    typer.echo(f"manifest -> {path}")


@app.command()
def parse() -> None:
    """Provenance + speaker-turn parse over the full corpus."""
    df = acquire.load_raw_frames()
    _write(pipeline.build_turns(df), INTERIM / "turns.parquet")


@app.command()
def extract(model: str = SPACY_MODEL_FULL) -> None:
    """Inline quote-attribution extraction over the full corpus."""
    from covered import ner

    df = acquire.load_raw_frames()
    nlp = ner.load_nlp(model)
    _write(pipeline.build_attributions(df, nlp), INTERIM / "attributions.parquet")


@app.command()
def hhi() -> None:
    """Compute the annual HHI series from the interim tables."""
    turns = pd.read_parquet(INTERIM / "turns.parquet")
    atts = pd.read_parquet(INTERIM / "attributions.parquet")
    annual = _annual_all(turns, atts)
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "hhi_annual.csv"
    annual.to_csv(out, index=False)
    typer.echo(f"wrote {len(annual)} rows -> {out}")


@app.command()
def make_figures() -> None:
    """Plot the HHI series and coverage diagnostic."""
    annual = pd.read_csv(TABLES / "hhi_annual.csv")
    figures.plot_hhi_series(annual, FIGURES / "hhi_series.png")
    typer.echo(f"figures -> {FIGURES}")


@app.command()
def pilot(
    start: str = "2014-06-18",
    end: str = "2015-12-31",
    model: str = "en_core_web_sm",
) -> None:
    """Run the whole chain on one date window and assert face validity."""
    from covered import ner

    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    df = acquire.load_raw_frames()
    air = pd.to_datetime(
        df[["year", "month", "date"]].rename(columns={"date": "day"}), errors="coerce"
    )
    mask = (air >= pd.Timestamp(start_d)) & (air <= pd.Timestamp(end_d))
    slice_df = df[mask].copy()
    typer.echo(f"pilot window {start}..{end}: {len(slice_df):,} segments")

    turns = pipeline.build_turns(slice_df)
    nlp = ner.load_nlp(model)
    atts = pipeline.build_attributions(slice_df, nlp)
    _write(turns, PROCESSED / "pilot_turns.parquet")
    _write(atts, PROCESSED / "pilot_attributions.parquet")

    report = check_face_validity(atts)
    for line in report:
        typer.echo(line)


def check_face_validity(atts: pd.DataFrame, top_k: int = 5) -> list[str]:
    """Assert the sitting president is among the top-k cited individuals per year.

    Returns a human-readable report; raises AssertionError on a hard miss, which
    signals broken extraction or entity resolution.
    """
    report: list[str] = []
    persons = atts[atts["entity_type"] == "PERSON"]
    for year, grp in persons.groupby("year"):
        yr = int(cast("int", year))
        expected = PRESIDENT_BY_YEAR.get(yr)
        if not expected:
            continue
        top = grp["canonical_id"].value_counts().head(top_k).index.tolist()
        ok = expected in top
        report.append(
            f"{yr}: expected '{expected}' in top{top_k} -> {'OK' if ok else 'MISS'} ({top})"
        )
        if not ok:
            raise AssertionError(
                f"face-validity failure {yr}: '{expected}' not in top{top_k} {top}"
            )
    return report


if __name__ == "__main__":
    app()
