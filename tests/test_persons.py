"""Entity form and name-gender readings.

Entity-form tests run on ``en_core_web_sm``, the model the test group installs.
It is weaker than the ``_lg`` model used in production, so the cases here are
ones any model should get; accuracy claims belong in the diagnostic scripts,
not in assertions that would fail on a model upgrade.
"""

from __future__ import annotations

import math

import pytest

from covered import persons, taxonomy
from covered.ner import load_nlp


@pytest.fixture(scope="module")
def nlp():
    return load_nlp("en_core_web_sm")


def test_a_personal_name_is_a_person(nlp) -> None:
    form, source = persons.entity_form("Barack Obama", nlp)
    assert (form, source) == ("person", "ner")


def test_an_outlet_is_an_organization(nlp) -> None:
    form, _ = persons.entity_form("The Washington Post", nlp)
    assert form == "organization"


def test_a_generic_plural_is_unnamed(nlp) -> None:
    """The category that breaks counting, and the reason bounds exist.

    "prosecutors" is a cited source that cannot be resolved to a countable
    person. Typing it as unknown would hide that; typing it as an organization
    would license counting it as one source.
    """
    form, _ = persons.entity_form("prosecutors", nlp)
    assert form == "unnamed"


def test_unidentified_speakers_come_from_the_list_not_the_model(nlp) -> None:
    """Transcript conventions are settled by the reference list, not spaCy."""
    form, source = persons.entity_form("UNIDENTIFIED MALE", nlp)
    assert (form, source) == ("unnamed", "nonperson_list")


def test_empty_input_is_unknown_not_unnamed(nlp) -> None:
    """No string is a different fact from a string naming no one."""
    assert persons.entity_form("", nlp) == ("unknown", "none")


def test_every_form_is_in_the_declared_vocabulary(nlp) -> None:
    probes = ["Barack Obama", "The Washington Post", "prosecutors", "", "CROWD", "IDF"]
    for probe in probes:
        form, _ = persons.entity_form(probe, nlp)
        assert form in taxonomy.ENTITY_FORMS, probe


def test_gender_reads_the_given_name() -> None:
    assert persons.gender("Kamala Harris").label == "female"
    assert persons.gender("Barack Obama").label == "male"


def test_gender_works_on_non_anglo_names() -> None:
    """The corpus is thick with Slavic and Hebrew names, so this is the case.

    The published benchmark warns of roughly tenfold error on Asian names; that
    limit is real and is why results must be split by name origin. It is not a
    reason to assume the model fails on every non-English name.
    """
    assert persons.gender("Volodymyr Zelenskyy").label == "male"
    assert persons.gender("Olga Smirnova").label == "female"


def test_an_unreadable_name_abstains_rather_than_guessing() -> None:
    reading = persons.gender("J.D. Vance")
    assert reading.label == "unknown"
    assert reading.sources == 0
    assert math.isnan(reading.p_female)
    assert reading.source == "none"


def test_the_probability_is_kept_alongside_the_label() -> None:
    """A thresholded label cannot distinguish ambiguous from merely rare."""
    reading = persons.gender("Kamala Harris")
    assert 0.5 < reading.p_female <= 1.0
    assert reading.sources > 0


def test_every_gender_is_in_the_declared_vocabulary() -> None:
    for name in ["Kamala Harris", "Barack Obama", "J.D. Vance", ""]:
        assert persons.gender(name).label in taxonomy.GENDERS, name
