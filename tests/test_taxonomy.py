"""Tests for the crossed source taxonomy."""

import pytest

from covered import taxonomy


def _label(role: str) -> taxonomy.SourceLabel:
    return taxonomy.classify(role)


# --- the crossing earns its keep -------------------------------------------
# These four share sectors pairwise but split on the epistemic axis, which is
# the whole justification for crossing rather than using a flat scheme.


def test_same_sector_splits_on_epistemic_role() -> None:
    lead = _label("LEAD PROSECUTOR")
    former = _label("FORMER FEDERAL PROSECUTOR")
    assert lead.sector == former.sector == "judicial_legal"
    assert lead.epistemic_role == "participant"
    assert former.standing == "former"
    assert former.epistemic_role != "participant"


def test_attorney_for_a_named_party_is_a_participant_not_a_pundit() -> None:
    pundit = _label("DEFENSE ATTORNEY")
    counsel = _label("ATTORNEY FOR GEORGE ZIMMERMAN")
    assert pundit.sector == counsel.sector == "judicial_legal"
    assert counsel.epistemic_role == "participant"
    assert pundit.epistemic_role == "commentator"


def test_lexicons_compose_across_axes() -> None:
    # Neither lexicon enumerates this pair; sector comes from one, role the other.
    label = _label("PENTAGON SPOKESMAN")
    assert (label.sector, label.epistemic_role) == ("military", "spokesperson")
    assert label.epistemic_source == "lexicon"


# --- standing ---------------------------------------------------------------


def test_former_officeholder_becomes_a_commentator() -> None:
    sitting = _label("SECRETARY OF STATE")
    former = _label("FORMER SECRETARY OF STATE")
    assert sitting.epistemic_role == "principal"
    assert sitting.standing == "current"
    assert former.standing == "former"
    assert former.epistemic_role == "commentator"
    assert former.epistemic_source == "standing_rule"


def test_former_credentialed_source_is_flagged_for_topic_resolution() -> None:
    # A credential does not lapse, so expert-vs-commentator here depends on the
    # segment topic. The label must say so rather than quietly pick one.
    label = _label("FORMER FBI SPECIAL AGENT")
    assert label.standing == "former"
    assert label.epistemic_source == "topic_needed"


def test_candidate_standing_is_detected() -> None:
    assert _label("(R) PRESIDENTIAL CANDIDATE").standing == "candidate"
    assert _label("GORE CAMPAIGN ATTORNEY").standing == "candidate"


# --- modifiers are not roles ------------------------------------------------


@pytest.mark.parametrize(
    ("role", "delivery"),
    [
        ("CNN CORRESPONDENT (voice-over)", "voice_over"),
        ("NANCY GRACE PRODUCER (via telephone)", "telephone"),
        ("UKRAINIAN PRESIDENT (through translator)", "translated"),
        ("SECRETARY OF STATE", "live"),
    ],
)
def test_delivery_is_lifted_out_of_the_role(role: str, delivery: str) -> None:
    assert _label(role).delivery == delivery


def test_stripping_collapses_otherwise_identical_roles() -> None:
    plain = _label("UKRAINIAN PRESIDENT")
    tagged = _label("UKRAINIAN PRESIDENT (through translator)")
    assert plain.role_clean == tagged.role_clean
    assert (plain.sector, plain.epistemic_role) == (
        tagged.sector,
        tagged.epistemic_role,
    )


def test_party_tag_is_lifted_out_of_the_role() -> None:
    label = _label("(D) PRESIDENT OF THE UNITED STATES")
    assert label.party == "D"
    assert label.sector == "government_executive"
    assert label.epistemic_role == "principal"


@pytest.mark.parametrize("role", ["(R) ARIZONA", "(D-NY) SENATOR", "(I) VERMONT"])
def test_real_party_tags_still_parse(role: str) -> None:
    assert _label(role).party in {"R", "D", "I"}


@pytest.mark.parametrize(
    "role", ["U.S. ARMY (RET.)", "LT. GEN. (RET.)", "GEN. WESLEY CLARK (RET.)"]
)
def test_retired_marker_is_not_mistaken_for_a_party_tag(role: str) -> None:
    # "(RET.)" once matched the party pattern as "R", which both invented a
    # Republican and erased the standing that makes a retired officer a
    # commentator rather than a serving spokesperson.
    label = _label(role)
    assert label.party == "none"
    assert label.standing == "former"


# --- polity -----------------------------------------------------------------


def test_polity_separates_us_from_foreign_and_local() -> None:
    assert _label("U.S. SECRETARY OF STATE").polity == "us_federal"
    assert _label("MAYOR").polity == "us_state_local"
    assert _label("UKRAINIAN PRESIDENT").polity == "foreign"
    # non-governmental sectors have no polity to report
    assert _label("ACTOR").polity == "na"


# --- the categories the old lexicon could not see ---------------------------


@pytest.mark.parametrize(
    ("role", "sector", "epistemic_role"),
    [
        ("MOTHER", "private_individual", "personal_experience"),
        ("NEIGHBOR", "private_individual", "eyewitness"),
        ("TORNADO SURVIVOR", "private_individual", "personal_experience"),
        ("MURDER DEFENDANT", "private_individual", "participant"),
        ("UKRAINIAN REFUGEE", "private_individual", "personal_experience"),
        ("REPUBLICAN STRATEGIST", "party_campaign", "commentator"),
        ("BROOKINGS INSTITUTION", "think_tank", "expert"),
        ("AMS METEOROLOGIST", "professional", "expert"),
        ("ACTOR", "entertainment_sport", "principal"),
    ],
)
def test_previously_unclassifiable_roles(
    role: str, sector: str, epistemic_role: str
) -> None:
    label = _label(role)
    assert (label.sector, label.epistemic_role) == (sector, epistemic_role)


# --- honest unknowns --------------------------------------------------------


def test_unmatched_role_is_unknown_not_other() -> None:
    label = _label("BAR-BE-QUE-HUT")
    assert label.sector == "unknown"
    assert label.epistemic_role == "unresolved"


def test_missing_role_does_not_guess() -> None:
    for empty in (None, "", "   "):
        label = taxonomy.classify(empty)
        assert label.sector == "unknown"
        assert label.epistemic_role == "unresolved"


def test_longest_keyword_wins_regardless_of_row_order() -> None:
    # "criminal defense attorney" and "attorney" both match; the specific one
    # must win, and it must not depend on where the rows sit in the file.
    assert (
        _label("CRIMINAL DEFENSE ATTORNEY").sector_rule == "criminal defense attorney"
    )
    assert _label("WHITE HOUSE PRESS SECRETARY").epistemic_rule == (
        "white house press secretary"
    )


# --- the office lives in the name, not the role -----------------------------
# CNN writes elected officials as "SEN. JOHN MCCAIN, (R) ARIZONA:". Reading the
# role clause alone leaves a bare state name and loses the whole category.


def test_elected_official_is_recovered_from_the_name_honorific() -> None:
    label = taxonomy.classify("(R) ARIZONA", name="SEN. JOHN MCCAIN")
    assert label.sector == "government_legislative"
    assert label.sector_source == "name_title"
    assert label.party == "R"


def test_name_honorific_distinguishes_chamber_from_statehouse() -> None:
    sen = taxonomy.classify("(D) NEW YORK", name="SEN. HILLARY CLINTON")
    gov = taxonomy.classify("(D) MARYLAND", name="GOV. PARRIS GLENDENING")
    assert sen.sector == "government_legislative"
    assert gov.sector == "government_subnational"


def test_party_plus_state_falls_back_to_legislative_without_a_name() -> None:
    label = taxonomy.classify("(R) TEXAS")
    assert label.sector == "government_legislative"
    assert label.sector_source == "party_state"  # flags the modal-case assumption


def test_an_explicit_role_always_beats_the_name_honorific() -> None:
    # Dr. Sanjay Gupta as a network correspondent is media, not a physician.
    label = taxonomy.classify("CHIEF MEDICAL CORRESPONDENT", name="DR. SANJAY GUPTA")
    assert label.sector_source == "role"


def test_bare_state_without_a_party_tag_is_not_promoted() -> None:
    # "TEXAS" alone is not evidence of elected office.
    assert taxonomy.classify("TEXAS").sector == "unknown"


# --- outlets identified by form, not by an enumerated name ------------------


@pytest.mark.parametrize(
    "role",
    [
        '"CHICAGO SUN-TIMES"',
        '"THE NEW YORKER"',
        '"WALL STREET JOURNAL"',
        "ANDREWSULLIVAN.COM",
        "HIPHOLLYWOOD.COM",
    ],
)
def test_quoted_or_domain_role_is_a_media_outlet(role: str) -> None:
    label = taxonomy.classify(role)
    assert label.sector == "media"
    assert label.sector_source == "form"


def test_a_keyword_match_beats_the_typographic_rule() -> None:
    # Quoted, but the role says what they do, so the keyword should win.
    label = taxonomy.classify('SYNDICATED COLUMNIST, "CHICAGO SUN-TIMES"')
    assert label.sector == "media"
    assert label.sector_source == "role"


def test_quoting_is_not_read_into_ordinary_roles() -> None:
    assert taxonomy.classify('HE SAID "NO"').sector_source != "form"


def test_every_label_uses_declared_vocabulary() -> None:
    roles = [
        "SECRETARY OF STATE",
        "ACTOR",
        "MOTHER",
        "BROOKINGS INSTITUTION",
        "BAR-BE-QUE-HUT",
        "FORMER PROSECUTOR",
    ]
    for r in roles:
        label = _label(r)
        assert label.sector in taxonomy.SECTORS
        assert label.epistemic_role in taxonomy.EPISTEMIC_ROLES
