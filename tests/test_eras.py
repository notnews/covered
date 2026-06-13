"""Tests for era assignment (HTML/text-format epochs of the CNN corpus)."""

from datetime import date

import pytest

from covered import eras


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2000, 1, 1), "era1"),  # corpus start
        (date(2002, 9, 16), "era1"),  # last day of old h-tag format
        (date(2002, 9, 17), "era2"),  # first cnnBodyText day
        (date(2014, 6, 17), "era2"),  # last cnnBodyText day
        (date(2014, 6, 18), "era3"),  # first modern-slug day
        (date(2025, 3, 15), "era3"),  # corpus end
    ],
)
def test_era_for_date_boundaries(day: date, expected: str) -> None:
    assert eras.era_for_date(day) == expected


def test_era_for_date_rejects_pre_corpus() -> None:
    with pytest.raises(ValueError, match="before the corpus start"):
        eras.era_for_date(date(1999, 12, 31))


def test_all_era_ids_known() -> None:
    assert eras.ERA_IDS == ("era1", "era2", "era3")
