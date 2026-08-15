"""The Wikidata tier: its cache, and the guard that decides a match.

Nothing here touches the network. ``resolve`` is cache-only by design, and the
matching rule -- the part that was actually wrong twice while being written --
is a pure function.
"""

from __future__ import annotations

import csv

import pytest

from covered import wikidata


def test_an_exact_label_matches() -> None:
    assert wikidata._label_matches("Adam Schiff", "adam schiff")


def test_a_middle_initial_is_tolerated() -> None:
    """Wikidata prefers the legal form where a transcript uses the everyday one."""
    assert wikidata._label_matches("James R. Clapper", "james clapper")
    assert wikidata._label_matches("Brendan F. Boyle", "brendan boyle")


def test_diacritics_are_folded() -> None:
    """The case where a stricter guard silently swapped in the wrong person.

    CNN writes "Aleksandar Vucic"; Wikidata labels him "Aleksandar Vučić".
    Comparing literally rejected the president of Serbia and accepted an
    unrelated namesake carrying a middle initial.
    """
    assert wikidata._label_matches("Aleksandar Vučić", "aleksandar vucic")
    assert wikidata._label_matches("Anders Åslund", "anders aslund")


def test_a_different_person_is_rejected() -> None:
    """Taking the first human search hit produced each of these."""
    assert not wikidata._label_matches("Jack Kelly", "john kelly")
    assert not wikidata._label_matches("Nina Appel", "nina schick")
    assert not wikidata._label_matches("Elizabeth Mansfield", "paula reid")


def test_an_unlabelled_item_is_rejected() -> None:
    """Unverifiable is treated as unmatched; the name model covers the gap."""
    assert not wikidata._label_matches("", "ryan nobles")


def test_a_surname_alone_does_not_match() -> None:
    assert not wikidata._label_matches("Schiff", "adam schiff")


def test_resolve_reads_the_cache_and_normalises_the_key(tmp_path, monkeypatch) -> None:
    path = tmp_path / "wikidata_persons.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# comment line must be skipped\n")
        writer = csv.DictWriter(fh, fieldnames=wikidata._FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "name_norm": "adam schiff",
                "qid": "Q350843",
                "label": "Adam Schiff",
                "gender": "male",
                "gender_qid": "Q6581097",
                "occupation": "Q82955",
            }
        )
    monkeypatch.setattr(wikidata, "cache_path", lambda: path)
    wikidata._cache.cache_clear()

    person = wikidata.resolve("Rep. Adam Schiff")
    assert person is not None
    assert person.gender == "male"
    assert person.qid == "Q350843"
    assert wikidata.resolve("Nobody At All") is None
    wikidata._cache.cache_clear()


def test_a_missing_cache_is_empty_not_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wikidata, "cache_path", lambda: tmp_path / "absent.csv")
    wikidata._cache.cache_clear()
    assert wikidata.resolve("Adam Schiff") is None
    wikidata._cache.cache_clear()


@pytest.mark.parametrize(("qid", "expected"), [("Q6581097", "male"), ("Q6581072", "female")])
def test_the_two_common_gender_values_map(qid, expected) -> None:
    assert wikidata._GENDER_QIDS[qid] == expected


def test_other_gender_values_are_not_coerced() -> None:
    """A P21 outside the pair is recorded, never rounded to the nearer bucket."""
    entity = {
        "claims": {"P21": [{"mainsnak": {"datavalue": {"value": {"id": "Q1097630"}}}}]},
        "labels": {"en": {"value": "Someone"}},
    }
    person = wikidata._person(entity, "Q1", "someone")
    assert person.gender == "unknown"
    assert person.gender_qid == "Q1097630", "the raw value survives for auditing"


def test_the_committed_cache_is_readable_and_uses_the_declared_vocabulary() -> None:
    from covered import taxonomy

    for person in wikidata._cache().values():
        assert person.gender in taxonomy.GENDERS
        assert person.qid.startswith("Q")
