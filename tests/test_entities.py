"""Tests for entity resolution (normalization, blocking, surname disambiguation)."""

import pytest

from covered import entities


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sen. John McCain (R-AZ)", "john mccain"),
        ("President Barack Obama", "barack obama"),
        ("Dr. Sanjay Gupta", "sanjay gupta"),
        ("COOPER", "cooper"),
        ("Vice President Joe Biden", "joe biden"),
        ("Rep. Nancy Pelosi, (D) California", "nancy pelosi"),
        ("Mr. James O'Brien", "james o'brien"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert entities.normalize_name(raw) == expected


def test_block_key_matches_on_surname() -> None:
    assert entities.block_key("john smith") == entities.block_key("smith")
    assert entities.block_key("jane smith") == entities.block_key("john smith")


def test_load_aliases_keys_are_normalized() -> None:
    aliases = entities.load_aliases()
    assert aliases["biden"] == "joe biden"
    # every key is already in normalized form (normalization is idempotent)
    assert all(k == entities.normalize_name(k) for k in aliases)


def test_resolve_uses_alias_table_for_head() -> None:
    out = entities.resolve_mentions(["Joe Biden", "President Biden", "Biden"])
    assert out == ["joe biden", "joe biden", "joe biden"]


def test_resolve_bare_surname_to_single_full_candidate() -> None:
    out = entities.resolve_mentions(["Jane Doe", "Ms. Doe", "Doe"])
    assert out == ["jane doe", "jane doe", "jane doe"]


def test_resolve_same_surname_collision_is_ambiguous() -> None:
    out = entities.resolve_mentions(["John Smith", "Jane Smith", "Smith"])
    assert out[0] == "john smith"
    assert out[1] == "jane smith"
    assert out[2] == "ambiguous:smith"


def test_resolve_standalone_bare_surname_keeps_itself() -> None:
    out = entities.resolve_mentions(["Kasparov"])
    assert out == ["kasparov"]
