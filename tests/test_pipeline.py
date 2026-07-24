"""End-to-end pipeline test on a synthetic 2-segment corpus (small spaCy model).

Proves measure (a), measure (b), entity resolution, and annual HHI compose into
the provenance-rich long tables and the headline series without gated data.
"""

import pandas as pd
import pytest

from covered import ner, pipeline


@pytest.fixture(scope="module")
def nlp():  # type: ignore[no-untyped-def]
    return ner.load_nlp("en_core_web_sm")


def _corpus() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "url": "http://transcripts.cnn.com/TRANSCRIPTS/2022.02.01/acd.01.html",
                "channel.name": "CNN",
                "program.name": "Anderson Cooper 360",
                "uid": "acd.01",
                "duration": "",
                "year": 2022,
                "month": 2,
                "date": 1,
                "time": "20:00",
                "timezone": "ET",
                "path": "2022.02.01/acd.01.html",
                "wordcount": 30,
                "subhead": "Economy",
                "text": (
                    "ANDERSON COOPER, CNN HOST: Tonight we discuss the economy. "
                    "Joe Biden said the plan works. "
                    "SENATOR JANE DOE, (D) NEW YORK: I disagree with the president."
                ),
            },
            {
                "url": "http://transcripts.cnn.com/TRANSCRIPTS/2022.02.02/acd.02.html",
                "channel.name": "CNN",
                "program.name": "Anderson Cooper 360",
                "uid": "acd.02",
                "duration": "",
                "year": 2022,
                "month": 2,
                "date": 2,
                "time": "20:00",
                "timezone": "ET",
                "path": "2022.02.02/acd.02.html",
                "wordcount": 25,
                "subhead": "Reform",
                "text": (
                    "ANDERSON COOPER, CNN HOST: More news tonight. "
                    "According to Barack Obama, reform is needed. "
                    "JOHN SMITH, EYEWITNESS: Indeed it is."
                ),
            },
        ]
    )


def test_build_turns_has_provenance_and_flags(nlp) -> None:  # type: ignore[no-untyped-def]
    turns = pipeline.build_turns(_corpus())
    assert len(turns) == 4
    # provenance is carried on every row
    assert {"uid", "url", "show_code", "year", "era_id", "staff_flag"} <= set(
        turns.columns
    )
    assert set(turns["show_code"]) == {"acd"}
    guests = turns[turns["staff_flag"] == "guest"]
    assert len(guests) == 2  # Jane Doe, John Smith


def test_build_attributions_persons_resolved(nlp) -> None:  # type: ignore[no-untyped-def]
    atts = pipeline.build_attributions(_corpus(), nlp)
    persons = atts[atts["entity_type"] == "PERSON"]
    assert "joe biden" in set(persons["canonical_id"])
    assert "barack obama" in set(persons["canonical_id"])
    # provenance present for audit
    assert {"uid", "char_start", "char_end", "sentence_text"} <= set(atts.columns)


def test_annual_hhi_speakers_external_only(nlp) -> None:  # type: ignore[no-untyped-def]
    turns = pipeline.build_turns(_corpus())
    series = pipeline.annual_hhi_speakers(turns, variant="external")
    row = series[series["year"] == 2022].iloc[0]
    assert row["n_distinct"] == 2  # two distinct guests
    assert row["measure"] == "speakers"
    assert row["variant"] == "external"


def test_speaker_mode_filter(nlp) -> None:  # type: ignore[no-untyped-def]
    turns = pipeline.build_turns(_corpus())
    # the synthetic corpus has no video clips -> all turns are live
    assert set(turns["source_mode"]) == {"live"}
    live = pipeline.annual_hhi_speakers(turns, variant="external", mode="live")
    clip = pipeline.annual_hhi_speakers(turns, variant="external", mode="clip")
    assert live.iloc[0]["variant"] == "external-live"
    assert live.iloc[0]["n_distinct"] == 2
    assert clip.empty  # no clip turns in this corpus


def test_annual_hhi_attributions_dedup_per_segment(nlp) -> None:  # type: ignore[no-untyped-def]
    atts = pipeline.build_attributions(_corpus(), nlp)
    series = pipeline.annual_hhi_attributions(atts)
    row = series[series["year"] == 2022].iloc[0]
    assert row["n_distinct"] == 2  # Biden, Obama
    assert row["measure"] == "attributions"
