"""Tests for reference-table loaders."""

from covered import provenance as prov


def test_load_show_map_returns_code_to_host() -> None:
    show_map = prov.load_show_map()
    assert show_map["acd"] == "Anderson Cooper"
    assert show_map["ampr"] == "Christiane Amanpour"
    # codes are normalized to lower-case
    assert all(code == code.lower() for code in show_map)
