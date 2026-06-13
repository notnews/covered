"""Tests for inline quote-attribution extraction (measure b).

Uses the small spaCy model for speed; production pins en_core_web_lg/trf.
Assertions are tolerant of model quirks (substring/field checks, not exact
offsets of model-produced spans), except the offset round-trip which must hold.
"""

import pytest

from covered import attribution, ner


@pytest.fixture(scope="module")
def nlp():  # type: ignore[no-untyped-def]
    return ner.load_nlp("en_core_web_sm")


def _sources(atts: list[attribution.Attribution]) -> list[str]:
    return [a.source_span.lower() for a in atts]


def test_subject_cue_person(nlp) -> None:  # type: ignore[no-untyped-def]
    atts = attribution.extract_attributions("Joe Biden said the economy is strong.", nlp)
    assert any("biden" in s for s in _sources(atts))
    hit = next(a for a in atts if "biden" in a.source_span.lower())
    assert hit.entity_type == "PERSON"
    assert hit.cue_verb == "say"


def test_inverted_attribution(nlp) -> None:  # type: ignore[no-untyped-def]
    atts = attribution.extract_attributions('"We will win," said Hillary Clinton.', nlp)
    assert any("clinton" in s for s in _sources(atts))


def test_according_to(nlp) -> None:  # type: ignore[no-untyped-def]
    atts = attribution.extract_attributions("According to Barack Obama, the plan worked.", nlp)
    hit = next(a for a in atts if "obama" in a.source_span.lower())
    assert hit.cue_verb == "according to"
    assert hit.pattern_id == "according_to"


def test_org_source_typed(nlp) -> None:  # type: ignore[no-untyped-def]
    atts = attribution.extract_attributions("The White House said it would respond.", nlp)
    hit = next(a for a in atts if "white house" in a.source_span.lower())
    assert hit.entity_type == "ORG"


def test_self_reference_excluded(nlp) -> None:  # type: ignore[no-untyped-def]
    atts = attribution.extract_attributions(
        "Joe Biden said hello to the crowd.", nlp, exclude_names={"joe biden"}
    )
    assert atts == []


def test_no_cue_no_attribution(nlp) -> None:  # type: ignore[no-untyped-def]
    assert attribution.extract_attributions("The weather is nice today.", nlp) == []


def test_offsets_map_back_to_source(nlp) -> None:  # type: ignore[no-untyped-def]
    text = "Joe Biden said the economy is strong. According to Barack Obama, it worked."
    atts = attribution.extract_attributions(text, nlp)
    assert atts
    for a in atts:
        assert text[a.char_start : a.char_end] == a.source_span
