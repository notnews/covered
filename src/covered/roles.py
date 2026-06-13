"""Tier-1 enrichment: derive office, party, and sitting-president flags.

All three signals are *deterministic* and mostly *transcript-internal*:

* :func:`classify_office` maps the raw speaker role to a coarse office category
  via a curated, order-sensitive lexicon (``office_lexicon.csv``).
* :func:`parse_party` reads party from the congressional ``(R-TX)``/``D-CALIF``
  abbreviation (high confidence) or a leading party word (lower confidence).
* :func:`president_on` / :func:`is_sitting_president` resolve the sitting US
  president for an air date from ``presidents.csv`` -- the basis for the
  president's-share-of-airtime figure.

Coverage is partial by design (most roles are not offices, most guests are not
party-tagged); callers should report the matched share rather than assume full
coverage.
"""

from __future__ import annotations

import csv
import functools
import re
from collections.abc import Iterable
from datetime import date

from covered.config import REFERENCE

__all__ = [
    "classify_office",
    "is_sitting_president",
    "parse_party",
    "president_on",
]

# Congressional party tag: ``(R-TX)``, ``D-CALIF.``, ``I-VT`` -- a letter, a
# hyphen, then a state abbreviation/name. The leading boundary keeps it from
# firing inside words. Highest-confidence party signal (elected officials).
_PARTY_ABBR = re.compile(r"(?:^|[\s(])([rdi])-[a-z]{2,}", re.IGNORECASE)

# Lower-confidence fallback: a bare party word anywhere in the role.
_PARTY_WORDS = (("republican", "R"), ("democratic", "D"), ("democrat", "D"), ("independent", "I"))


@functools.lru_cache(maxsize=1)
def _office_lexicon() -> tuple[tuple[str, str], ...]:
    """Ordered ``(keyword, office)`` pairs; first substring hit wins."""
    out: list[tuple[str, str]] = []
    with (REFERENCE / "office_lexicon.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(_decomment(fh)):
            kw = (row.get("keyword") or "").strip().lower()
            office = (row.get("office") or "").strip().lower()
            if kw and office:
                out.append((kw, office))
    return tuple(out)


@functools.lru_cache(maxsize=1)
def _presidents() -> tuple[tuple[str, str, date, date | None], ...]:
    """``(canonical_id, party, start, end)`` rows, end ``None`` if still serving."""
    out: list[tuple[str, str, date, date | None]] = []
    with (REFERENCE / "presidents.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(_decomment(fh)):
            cid = (row.get("canonical_id") or "").strip().lower()
            party = (row.get("party") or "").strip().upper()
            start = (row.get("start") or "").strip()
            end = (row.get("end") or "").strip()
            if cid and start:
                out.append(
                    (
                        cid,
                        party,
                        date.fromisoformat(start),
                        date.fromisoformat(end) if end else None,
                    )
                )
    return tuple(out)


def _decomment(lines: Iterable[str]) -> list[str]:
    """Drop ``#`` comment and blank lines so csv.DictReader sees the header first."""
    return [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def classify_office(role: str | None) -> str:
    """Return a coarse office category for ``role``, or ``""`` if none matches.

    Matched case-insensitively against the curated lexicon in file order, so
    specific phrases (``vice president``) win over the broad keyword they contain
    (``president``). Non-string input (e.g. a pandas ``NaN``) yields ``""``.
    """
    if not isinstance(role, str) or not role:
        return ""
    low = role.lower()
    for keyword, office in _office_lexicon():
        if keyword in low:
            return office
    return ""


def parse_party(role: str | None) -> str | None:
    """Return ``"R"``/``"D"``/``"I"`` for ``role``, or ``None``.

    Prefers the high-confidence congressional abbreviation (``(R-TX)``); falls
    back to a party word (``REPUBLICAN STRATEGIST``), which is weaker evidence of
    actual party membership. Non-string input (e.g. a pandas ``NaN``) yields ``None``.
    """
    if not isinstance(role, str) or not role:
        return None
    m = _PARTY_ABBR.search(role)
    if m:
        return m.group(1).upper()
    low = role.lower()
    for word, party in _PARTY_WORDS:
        if word in low:
            return party
    return None


def president_on(day: date | None) -> str | None:
    """Canonical id of the US president sitting on ``day`` (``None`` if unknown).

    Terms are inclusive of the inauguration date and exclusive of the successor's.
    """
    if day is None:
        return None
    for cid, _party, start, end in _presidents():
        if start <= day and (end is None or day < end):
            return cid
    return None


def is_sitting_president(canonical_id: object, day: date | None) -> bool:
    """True if ``canonical_id`` is the person sitting as president on ``day``."""
    pres = president_on(day)
    return pres is not None and str(canonical_id).strip().lower() == pres
