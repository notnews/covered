"""Tests for the CSV schema validation."""

import pandas as pd

from covered import schema


def _valid_row() -> dict[str, object]:
    return {
        "url": "http://transcripts.cnn.com/TRANSCRIPTS/2022.02.01/acd.01.html",
        "channel.name": "CNN",
        "program.name": "Anderson Cooper 360 Degrees",
        "uid": "acd.01",
        "duration": "",
        "year": 2022,
        "month": 2,
        "date": 1,
        "time": "20:00",
        "timezone": "ET",
        "path": "2022.02.01/acd.01.html",
        "wordcount": 5000,
        "subhead": "Topic",
        "text": "WOLF BLITZER, CNN ANCHOR: Hello.",
    }


def test_valid_frame_passes() -> None:
    df = pd.DataFrame([_valid_row()])
    out = schema.validate_csv(df)
    assert len(out) == 1


def test_blank_wordcount_allowed() -> None:
    row = _valid_row()
    row["wordcount"] = None
    df = pd.DataFrame([row])
    assert len(schema.validate_csv(df)) == 1


def test_out_of_range_dates_coerced_to_na() -> None:
    # dirty numerics (month 13, year 3007) become NA rather than crashing the run
    bad = _valid_row()
    bad["month"] = 13
    bad["year"] = 3007
    out = schema.validate_csv(pd.DataFrame([bad, _valid_row()]))
    assert pd.isna(out["month"].iloc[0])
    assert pd.isna(out["year"].iloc[0])
    assert out["year"].iloc[1] == 2022
