"""Tests for CLI helpers that don't require the gated corpus."""

import pandas as pd
import pytest

from covered import cli


def _atts(canonical_ids: list[str], year: int = 2022) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [year] * len(canonical_ids),
            "entity_type": ["PERSON"] * len(canonical_ids),
            "canonical_id": canonical_ids,
        }
    )


def test_face_validity_passes_when_president_dominates() -> None:
    atts = _atts(["joe biden"] * 5 + ["someone else", "another"])
    report = cli.check_face_validity(atts)
    assert any("2022" in line and "OK" in line for line in report)


def test_face_validity_raises_on_miss() -> None:
    atts = _atts(["random a", "random b", "random c", "random d", "random e", "random f"])
    with pytest.raises(AssertionError, match="face-validity failure"):
        cli.check_face_validity(atts)
