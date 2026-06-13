"""Tests for provenance parsing (slug -> show_code/segment, plus audit fields)."""

from datetime import date

import pytest

from covered import provenance as prov


@pytest.mark.parametrize(
    ("uid", "expected"),
    [
        ("acd.01", ("acd", 1)),  # modern show.segment
        ("acd", ("acd", None)),  # no segment
        ("lkl.00", ("lkl", 0)),  # zero-padded segment
        ("ACD.02", ("acd", 2)),  # case-normalized
        ("cnni.cf.03", ("cnni", 3)),  # multi-token code, trailing numeric segment
        ("", (None, None)),  # empty
    ],
)
def test_parse_uid(uid: str, expected: tuple[str | None, int | None]) -> None:
    assert prov.parse_uid(uid) == expected


def test_parse_url_date_modern() -> None:
    url = "http://transcripts.cnn.com/TRANSCRIPTS/2022.02.01/acd.01.html"
    assert prov.parse_url_date(url) == date(2022, 2, 1)


def test_parse_url_date_missing() -> None:
    assert prov.parse_url_date("http://transcripts.cnn.com/TRANSCRIPTS/index.html") is None


def test_parse_provenance_full_row() -> None:
    row = {
        "url": "http://transcripts.cnn.com/TRANSCRIPTS/2022.02.01/acd.01.html",
        "channel.name": "CNN",
        "program.name": "Anderson Cooper 360 Degrees",
        "uid": "acd.01",
        "path": "2022.02.01/acd.01.html",
        "year": 2022,
        "month": 2,
        "date": 1,
        "time": "20:00",
        "timezone": "ET",
        "subhead": "Russia-Ukraine Crisis",
    }
    p = prov.parse_provenance(row, show_map={"acd": "Anderson Cooper 360"})

    assert p.uid == "acd.01"
    assert p.show_code == "acd"
    assert p.segment_index == 1
    assert p.air_date == date(2022, 2, 1)
    assert p.url_date == date(2022, 2, 1)
    assert p.era_id == "era3"
    assert p.program_name == "Anderson Cooper 360 Degrees"
    assert p.headline == "Anderson Cooper 360 Degrees"
    assert p.subhead == "Russia-Ukraine Crisis"
    assert p.host == "Anderson Cooper 360"
    assert p.channel_name == "CNN"


def test_parse_provenance_unknown_show_code_has_no_host() -> None:
    row = {
        "url": "http://transcripts.cnn.com/TRANSCRIPTS/2001.05.10/xyz.01.html",
        "program.name": "Some Program",
        "uid": "xyz.01",
        "path": "2001.05.10/xyz.01.html",
        "year": 2001,
        "month": 5,
        "date": 10,
    }
    p = prov.parse_provenance(row, show_map={"acd": "Anderson Cooper 360"})
    assert p.host is None
    assert p.era_id == "era1"
