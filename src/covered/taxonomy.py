"""Crossed source taxonomy: what kind of source a speaker is.

A ``NAME, ROLE:`` role clause does several jobs at once -- it names an
occupation (``DEFENSE ATTORNEY``), an office (``SECRETARY OF STATE``), a bare
organisation (``BROOKINGS INSTITUTION``), or a relation to the story
(``FRIEND OF GEORGE ZIMMERMAN``) -- and carries delivery and party tags that
are not roles at all. Flattening that into one label is what makes existing
source typologies lossy, so this module separates it into crossed axes:

* **sector** -- the institutional home (:data:`SECTORS`)
* **epistemic_role** -- why they are cited (:data:`EPISTEMIC_ROLES`)
* **standing** -- current, former, or seeking office
* **polity** -- whose government, where the sector is governmental
* **delivery** -- live, voice-over, telephone, translated

The pay-off is pairs a flat scheme cannot express: ``LEAD PROSECUTOR`` and
``FORMER FEDERAL PROSECUTOR`` share a sector but are participant and
commentator respectively.

Sector and epistemic role are matched by two independent lexicons under
``data/reference/``, each longest-keyword-wins, so ``PENTAGON SPOKESMAN``
composes to military x spokesperson without either lexicon enumerating the
pair. Longest-match (rather than first-hit-in-file-order, as
``office_lexicon.csv`` uses) keeps row order irrelevant, so adding a keyword
cannot silently shadow an existing one.
"""

from __future__ import annotations

import csv
import functools
import re
from dataclasses import dataclass

from covered.config import REFERENCE

__all__ = [
    "EPISTEMIC_ROLES",
    "SECTORS",
    "SECTOR_DEFAULT_ROLE",
    "SourceLabel",
    "classify",
    "strip_modifiers",
]

SECTORS: tuple[str, ...] = (
    "government_executive",
    "government_legislative",
    "government_subnational",
    "judicial_legal",
    "law_enforcement",
    "military",
    "party_campaign",
    "business",
    "academic",
    "think_tank",
    "nonprofit_advocacy",
    "labor_union",
    "religious",
    "media",
    "professional",
    "entertainment_sport",
    "private_individual",
    "unknown",
)

EPISTEMIC_ROLES: tuple[str, ...] = (
    "principal",
    "spokesperson",
    "expert",
    "commentator",
    "participant",
    "eyewitness",
    "personal_experience",
    "popular_opinion",
    "unresolved",
)

# Where the epistemic lexicon is silent, fall back on what the sector implies.
# Every one of these is a judgement call, so classify() tags the result
# epistemic_source="sector_default" and the census reports that share.
SECTOR_DEFAULT_ROLE: dict[str, str] = {
    "government_executive": "principal",
    "government_legislative": "principal",
    "government_subnational": "principal",
    "judicial_legal": "commentator",
    "law_enforcement": "spokesperson",
    "military": "expert",
    "party_campaign": "commentator",
    "business": "principal",
    "academic": "expert",
    "think_tank": "expert",
    "nonprofit_advocacy": "spokesperson",
    "labor_union": "spokesperson",
    "religious": "spokesperson",
    "media": "commentator",
    "professional": "expert",
    "entertainment_sport": "principal",
    "private_individual": "personal_experience",
    "unknown": "unresolved",
}

# Sectors whose knowledge is a credential rather than an appointment: it does
# not lapse when the person leaves the post, so "former" does not by itself
# turn them into a pundit. Which of expert/commentator applies then depends on
# whether the segment topic is their domain -- not knowable from the role
# string, so classify() marks those "topic_needed" for later resolution.
_CREDENTIALED = frozenset(
    {
        "academic",
        "professional",
        "judicial_legal",
        "law_enforcement",
        "military",
        "think_tank",
    }
)

# Roles held rather than earned: out of the post is out of the standing, so a
# former holder speaking on air is commenting, not speaking for the institution.
# "participant" belongs here too -- having had a direct role in past events is
# not having one in the events now being reported.
_POSITIONAL_ROLES = frozenset({"principal", "spokesperson", "participant"})

_STANDING_PATTERNS: tuple[tuple[str, str], ...] = (
    ("candidate", r"\bcandidate\b|\bnominee\b|\bcampaign\b|\brunning for\b"),
    ("former", r"\bformer\b|\bex-|\(ret\.?\)|\bretired\b|\boutgoing\b"),
)

_DELIVERY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("voice_over", r"\(\s*voice[\s-]?over\s*\)"),
    (
        "telephone",
        r"\((?:via|by|on)\s+(?:the\s+)?(?:telephone|phone)\)|\bvia telephone\b",
    ),
    ("translated", r"\(\s*through (?:translator|interpreter)\s*\)"),
    ("taped", r"\(\s*(?:on tape|taped|recorded)\s*\)"),
)

# A party tag is a bare R/D/I, optionally with a two-letter state: "(R)",
# "(D-NY)". Deliberately narrow and case-sensitive -- an earlier permissive
# version matched "(RET.)" as party "R", which silently turned every retired
# officer into a sitting Republican and destroyed their "former" standing.
_PARTY_TAG = re.compile(r"\(\s*([RDI])(?:\s*[-–]\s*[A-Z]{2})?\s*\)")  # noqa: RUF001

_INTERNATIONAL = re.compile(
    r"\bu\.?n\.?\b|united nations|\bnato\b|european union|world bank"
    r"|\bimf\b|world health organization|\bwho\b|\beu\b",
    re.IGNORECASE,
)
_US_FEDERAL = re.compile(
    r"\bu\.?s\.?\b|united states|american|white house|pentagon|congress"
    r"|federal|capitol hill",
    re.IGNORECASE,
)
_US_LOCAL = re.compile(
    r"\bcounty\b|\bcity\b|\bstate of\b|\bmayor\b|\bgovernor\b|\bmunicipal\b",
    re.IGNORECASE,
)
# An explicit list, not a suffix heuristic: -ian/-ish/-ese/-i as a pattern
# matches "civilian", "publish", "chinese" and "deputy" alike, which would put
# most of the corpus in "foreign". Better to miss a nationality and report
# "unknown" than to invent one.
_FOREIGN = re.compile(
    r"\b(?:afghan|african|albanian|algerian|arab|argentin\w*|armenian|australian"
    r"|austrian|bangladeshi|belarusian|belgian|bolivian|bosnian|brazilian|british"
    r"|bulgarian|burmese|cambodian|canadian|chilean|chinese|colombian|croatian"
    r"|cuban|czech|danish|dutch|egyptian|english|estonian|ethiopian|filipino"
    r"|finnish|french|georgian|german|ghanaian|greek|haitian|hungarian|icelandic"
    r"|indian|indonesian|iranian|iraqi|irish|israeli|italian|jamaican|japanese"
    r"|jordanian|kazakh|kenyan|korean|kosovar|kurdish|kuwaiti|latvian|lebanese"
    r"|liberian|libyan|lithuanian|malaysian|mexican|moroccan|nepalese|nigerian"
    r"|norwegian|pakistani|palestinian|peruvian|polish|portuguese|qatari|romanian"
    r"|russian|rwandan|saudi|scottish|serbian|singaporean|slovak|slovenian|somali"
    r"|spanish|sudanese|swedish|swiss|syrian|taiwanese|thai|tunisian|turkish"
    r"|ugandan|ukrainian|uzbek|venezuelan|vietnamese|welsh|yemeni|zimbabwean)\b",
    re.IGNORECASE,
)
_GOVERNMENTAL = frozenset(
    {
        "government_executive",
        "government_legislative",
        "government_subnational",
        "judicial_legal",
        "law_enforcement",
        "military",
    }
)

# Honorifics carried in the speaker NAME rather than the role clause, e.g.
# "SEN. JOHN MCCAIN, (R) ARIZONA:" -- the office is in the name and the role
# clause is only party and state. Consulted only when the role string itself
# yields no sector, so an explicit role always wins.
_NAME_TITLES: tuple[tuple[str, str, str], ...] = (
    (r"^sen\.|^senator\b", "government_legislative", "principal"),
    (
        r"^rep\.|^representative\b|^congressman\b|^congresswoman\b",
        "government_legislative",
        "principal",
    ),
    (r"^gov\.|^governor\b", "government_subnational", "principal"),
    (r"^mayor\b", "government_subnational", "principal"),
    (r"^judge\b|^justice\b|^chief justice\b", "judicial_legal", "principal"),
    (r"^amb\.|^ambassador\b", "government_executive", "principal"),
    (r"^president\b|^vice president\b", "government_executive", "principal"),
    (r"^sec\.|^secretary\b", "government_executive", "principal"),
    (
        r"^gen\.|^lt\. gen\.|^maj\. gen\.|^brig\. gen\.|^adm\.|^col\.|^capt\."
        r"|^maj\.|^sgt\.|^lt\.|^cmdr\.",
        "military",
        "expert",
    ),
    (r"^officer\b|^det\.|^detective\b|^sheriff\b", "law_enforcement", "spokesperson"),
    (
        r"^rev\.|^reverend\b|^bishop\b|^rabbi\b|^imam\b|^father\b",
        "religious",
        "spokesperson",
    ),
    (r"^prof\.|^professor\b", "academic", "expert"),
    (r"^dr\.|^doctor\b", "professional", "expert"),
)

# A role clause that is nothing but a US state, once the party tag is removed,
# is CNN's format for an elected official from that state. The chamber is not
# recoverable from it -- that lives in the name honorific -- so this is only
# the fallback when the name gives nothing.
_US_STATES = frozenset(
    {
        "alabama",
        "alaska",
        "arizona",
        "arkansas",
        "california",
        "colorado",
        "connecticut",
        "delaware",
        "florida",
        "georgia",
        "hawaii",
        "idaho",
        "illinois",
        "indiana",
        "iowa",
        "kansas",
        "kentucky",
        "louisiana",
        "maine",
        "maryland",
        "massachusetts",
        "michigan",
        "minnesota",
        "mississippi",
        "missouri",
        "montana",
        "nebraska",
        "nevada",
        "ohio",
        "oklahoma",
        "oregon",
        "pennsylvania",
        "tennessee",
        "texas",
        "utah",
        "vermont",
        "virginia",
        "washington",
        "wisconsin",
        "wyoming",
        "new hampshire",
        "new jersey",
        "new mexico",
        "new york",
        "north carolina",
        "north dakota",
        "rhode island",
        "south carolina",
        "south dakota",
        "west virginia",
        "district of columbia",
        "puerto rico",
    }
)


@dataclass(frozen=True, slots=True)
class SourceLabel:
    """What kind of source a speaker is, on crossed axes."""

    sector: str
    epistemic_role: str
    standing: str  # "current" | "former" | "candidate"
    polity: str  # "us_federal" | "us_state_local" | "foreign" | "international" | "na"
    delivery: str  # "live" | "voice_over" | "telephone" | "translated" | "taped"
    party: str  # "R" | "D" | "I" | "none"
    role_clean: str  # role text after modifiers were stripped
    sector_rule: str  # keyword that set the sector ("" if none matched)
    sector_source: str  # "role" | "name_title" | "party_state" | "none"
    epistemic_rule: str  # keyword that set the epistemic role ("" if none)
    epistemic_source: (
        str  # "lexicon" | "standing_rule" | "sector_default" | "topic_needed"
    )


def _load_lexicon(filename: str, value_column: str) -> tuple[tuple[str, str], ...]:
    """Load a keyword lexicon, sorted longest-keyword-first for greedy matching."""
    path = REFERENCE / filename
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            kw = (row.get("keyword") or "").strip().lower()
            val = (row.get(value_column) or "").strip().lower()
            if kw and val:
                rows.append((kw, val))
    rows.sort(key=lambda kv: len(kv[0]), reverse=True)
    return tuple(rows)


@functools.lru_cache(maxsize=1)
def _sector_lexicon() -> tuple[tuple[str, str], ...]:
    return _load_lexicon("sector_lexicon.csv", "sector")


@functools.lru_cache(maxsize=1)
def _epistemic_lexicon() -> tuple[tuple[str, str], ...]:
    return _load_lexicon("epistemic_lexicon.csv", "epistemic_role")


@functools.lru_cache(maxsize=1)
def _role_dictionary() -> dict[str, tuple[str, str]]:
    """Frozen per-string labels for roles no rule reaches.

    The keyword lexicons generalise; a long tail of role strings does not --
    ``PRINCESS DIANA'S LOVER 1986-91``, ``DIRECTOR EMERITUS, COLUMBUS ZOO``.
    Those are labelled once, offline, by an LLM working from the same codebook
    (``scripts/label_roles_llm.py``), and the result is committed here and
    applied by exact match.

    Three properties make this safe to use in a measurement:

    * the model never runs in the production path -- the committed CSV does, so
      the pipeline stays deterministic and reproducible without an API key;
    * it is consulted only where the keyword rules yield nothing, so the
      auditable layer always wins;
    * labels from it are tagged ``dictionary``, so every table can report what
      the tier contributed and validation can score it separately.

    Missing file is not an error: the dictionary is an optional enrichment.
    """
    path = REFERENCE / "role_dictionary.csv"
    if not path.exists():
        return {}
    out: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            key = (row.get("role_raw") or "").strip().lower()
            sector = (row.get("sector") or "").strip().lower()
            role = (row.get("epistemic_role") or "").strip().lower()
            if key and sector in SECTORS:
                out[key] = (sector, role if role in EPISTEMIC_ROLES else "")
    return out


@functools.lru_cache(maxsize=1)
def _outlets() -> frozenset[str]:
    """Curated news outlets, matched as a whole role clause."""
    path = REFERENCE / "news_outlets.txt"
    return frozenset(
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


def strip_modifiers(role: str) -> tuple[str, str, str]:
    """Split a raw role string into ``(role_clean, delivery, party)``.

    Delivery tags (``(voice-over)``, ``(via telephone)``, ``(through
    translator)``) describe how a speaker reached air, and party tags
    (``(R)``, ``(D)``) describe affiliation. Neither says what kind of source
    the person is, and leaving them inline fragments otherwise identical role
    strings, so both are lifted into their own fields.
    """
    delivery = "live"
    text = role
    for name, pattern in _DELIVERY_PATTERNS:
        stripped = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        if stripped != text:
            delivery = name
            text = stripped

    party = "none"
    m = _PARTY_TAG.search(text)
    if m:
        party = m.group(1).upper()
        text = _PARTY_TAG.sub(" ", text)

    return re.sub(r"\s+", " ", text).strip(" ,;-"), delivery, party


def _media_by_form(role_clean: str) -> str:
    """Detect a media outlet from the FORM of the role string, not a keyword.

    CNN sets an outlet name in quotes when a visiting journalist's affiliation is
    the whole role -- ``"CHICAGO SUN-TIMES"``, ``"THE NEW YORKER"`` -- and web
    outlets appear as bare domains. Both generalise typographically, without
    anyone enumerating outlet names.

    Often enough, though, the name is written bare: ``THE NEW YORK TIMES``,
    ``WALL STREET JOURNAL``, ``BLEACHER REPORT``. Those need a list, so a
    curated one is matched as the *whole* clause -- never as a substring, or
    "TIMES" would swallow unrelated roles.
    """
    text = role_clean.strip()
    if len(text) > 2 and text.startswith('"') and text.rstrip(";").endswith('"'):
        return "quoted_outlet"
    if re.search(r"\b[\w-]+\.(?:com|org|net|tv|news)\b", text, re.IGNORECASE):
        return "domain"
    bare = re.sub(r"^the\s+", "", text.lower().strip(" \"';,.")).strip()
    if bare in _outlets():
        return "named_outlet"
    return ""


@functools.lru_cache(maxsize=4096)
def _kw_pattern(kw: str) -> re.Pattern[str]:
    r"""Whole-word matcher for one keyword.

    Plain substring matching silently mislabels: "king" fires inside
    "BROOKINGS INSTITUTION", "dea" inside "IDEA" and "CAR DEALER". Lookarounds
    rather than ``\b`` because keywords may begin or end with punctuation
    ("sen.", "co-founder"), where ``\b`` sits in the wrong place.
    """
    return re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)")


def _match(lexicon: tuple[tuple[str, str], ...], text: str) -> tuple[str, str]:
    """Return ``(value, keyword)`` for the longest keyword occurring in ``text``."""
    for kw, val in lexicon:  # pre-sorted longest-first
        if _kw_pattern(kw).search(text):
            return val, kw
    return "", ""


def _polity(text: str, sector: str) -> str:
    if sector not in _GOVERNMENTAL:
        return "na"
    if _INTERNATIONAL.search(text):
        return "international"
    if _US_LOCAL.search(text):
        return "us_state_local"
    if _US_FEDERAL.search(text):
        return "us_federal"
    return "foreign" if _FOREIGN.search(text) else "unknown"


def _standing(text: str) -> str:
    for name, pattern in _STANDING_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return name
    return "current"


def _from_name_title(name: str) -> tuple[str, str, str]:
    """Return ``(sector, epistemic_role, matched_pattern)`` from a name honorific."""
    low = name.strip().lower()
    for pattern, sector, role in _NAME_TITLES:
        if re.search(pattern, low):
            return sector, role, pattern
    return "", "", ""


def classify(role_raw: str | None, name: str | None = None) -> SourceLabel:
    """Classify one speaker onto the crossed axes.

    ``name`` is the speaker label, consulted only when the role clause yields no
    sector. CNN writes elected officials as ``SEN. JOHN MCCAIN, (R) ARIZONA:`` --
    the office sits in the name and the role clause is only party and state, so
    without the name those turns are unclassifiable.

    A missing role and an uninformative name yield an all-unknown label rather
    than a guess, so unclassified turns stay countable instead of inflating a
    residual bucket.
    """
    if not role_raw or not str(role_raw).strip():
        sector, epistemic_role, rule = _from_name_title(name or "")
        return SourceLabel(
            sector=sector or "unknown",
            epistemic_role=epistemic_role or "unresolved",
            standing="current",
            polity="na",
            delivery="live",
            party="none",
            role_clean="",
            sector_rule=rule,
            sector_source="name_title" if sector else "none",
            epistemic_rule="",
            epistemic_source="name_title" if sector else "sector_default",
        )

    role_clean, delivery, party = strip_modifiers(str(role_raw))
    low = role_clean.lower()

    sector, sector_rule = _match(_sector_lexicon(), low)
    sector_source = "role" if sector else "none"
    epistemic_role, epistemic_rule = _match(_epistemic_lexicon(), low)
    epistemic_source = "lexicon" if epistemic_role else "sector_default"

    if not sector:
        # No keyword matched: try the typographic outlet convention, then the
        # name honorific, then the party-tag-plus-state elected-official format.
        form = _media_by_form(role_clean)
        n_sector, n_role, n_rule = _from_name_title(name or "")
        if form:
            # A speaker introduced by outlet alone is a visiting journalist, and
            # the form settles the epistemic role as well as the sector -- the
            # same judgement the "journalist" keyword makes, reached
            # typographically rather than lexically.
            sector, sector_rule, sector_source = "media", form, "form"
            if not epistemic_role:
                epistemic_role, epistemic_source = "commentator", "form"
        elif n_sector:
            sector, sector_rule, sector_source = n_sector, n_rule, "name_title"
            if not epistemic_role:
                epistemic_role, epistemic_source = n_role, "name_title"
        elif party != "none" and low in _US_STATES:
            # Chamber is not recoverable here; legislative is the modal case and
            # sector_source records that this cell rests on that assumption.
            sector, sector_rule, sector_source = (
                "government_legislative",
                low,
                "party_state",
            )
        else:
            # Last resort: the frozen per-string dictionary, for role strings no
            # rule generalises to. Reached only when every rule above declined.
            hit = _role_dictionary().get(low)
            if hit:
                sector, sector_rule, sector_source = hit[0], low, "dictionary"
                if not epistemic_role and hit[1]:
                    epistemic_role, epistemic_source = hit[1], "dictionary"
    sector = sector or "unknown"
    if not epistemic_role:
        epistemic_role = SECTOR_DEFAULT_ROLE.get(sector, "unresolved")

    standing = _standing(role_clean)
    # Leaving a post ends the standing that made someone a principal or a
    # spokesperson. Where the sector's authority is a credential instead, the
    # expert/commentator split turns on whether the segment topic is their
    # domain, which the role string cannot say -- flag it rather than guess.
    if standing == "former" and epistemic_role in _POSITIONAL_ROLES:
        if sector in _CREDENTIALED:
            epistemic_role, epistemic_source = "expert", "topic_needed"
        else:
            epistemic_role, epistemic_source = "commentator", "standing_rule"
    elif (
        standing == "former" and sector in _CREDENTIALED and epistemic_role == "expert"
    ):
        epistemic_source = "topic_needed"

    return SourceLabel(
        sector=sector,
        epistemic_role=epistemic_role,
        standing=standing,
        polity=_polity(role_clean, sector),
        delivery=delivery,
        party=party,
        role_clean=role_clean,
        sector_rule=sector_rule,
        sector_source=sector_source,
        epistemic_rule=epistemic_rule,
        epistemic_source=epistemic_source,
    )
