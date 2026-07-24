"""Tests for Tier-1 office/party/president enrichment (real reference tables)."""

from datetime import date

import pytest

from covered import roles


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("PRESIDENT OF THE UNITED STATES", "president"),
        (
            "VICE PRESIDENT OF THE UNITED STATES",
            "vice_president",
        ),  # specific beats "president"
        ("WHITE HOUSE PRESS SECRETARY", "press_secretary"),  # beats "secretary of"
        ("U.S. SECRETARY OF STATE", "cabinet_secretary"),
        ("U.S. SENATE MEMBER", "senator"),
        ("SENATOR (D-CA)", "senator"),
        ("REPUBLICAN PRES. CANDIDATE", "candidate"),
        ("BRITISH PRIME MINISTER", "diplomat"),
        ("ACTOR", "entertainment"),
        ("AMS METEOROLOGIST", ""),  # unmatched -> empty
        ("", ""),
        (None, ""),
    ],
)
def test_classify_office(role: str | None, expected: str) -> None:
    assert roles.classify_office(role) == expected


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("SENATOR (D-CA)", "D"),
        ("REP. (R-TX)", "R"),
        ("SENATOR, I-VT", "I"),
        ("D-CALIF.", "D"),
        ("REPUBLICAN STRATEGIST", "R"),  # word fallback
        ("DEMOCRATIC POLLSTER", "D"),
        ("ACTOR", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_party(role: str | None, expected: str | None) -> None:
    assert roles.parse_party(role) == expected


def test_parse_party_abbrev_beats_word() -> None:
    # A congressional abbreviation should win even if a party word is also present.
    assert roles.parse_party("DEMOCRAT (R-XX)") == "R"


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2000, 6, 1), "bill clinton"),
        (date(2001, 1, 20), "george w. bush"),  # inauguration day -> successor
        (date(2001, 1, 19), "bill clinton"),
        (date(2013, 1, 1), "barack obama"),
        (date(2018, 1, 1), "donald trump"),
        (date(2022, 1, 1), "joe biden"),
        (date(2025, 3, 1), "donald trump"),  # second term, open-ended
        (date(1990, 1, 1), None),  # before corpus
        (None, None),
    ],
)
def test_president_on(day: date | None, expected: str | None) -> None:
    assert roles.president_on(day) == expected


def test_is_sitting_president() -> None:
    assert roles.is_sitting_president("Donald Trump", date(2019, 5, 1)) is True
    assert roles.is_sitting_president("barack obama", date(2019, 5, 1)) is False
    assert roles.is_sitting_president("donald trump", None) is False
