"""Measure (a): parse ``NAME, ROLE:`` speaker turns from transcript text.

The parser finds speaker-label offsets across the whole blob (line-break
structure differs by era), builds a per-transcript roster so bare-surname
continuation turns resolve to the speaker's full identity, and classifies each
speaker as CNN staff, external guest, a non-person sentinel, or unknown.

Character offsets of every utterance are preserved so each turn can be audited
against the source ``text`` field.
"""

from __future__ import annotations

import csv
import functools
import re
from dataclasses import dataclass

from covered.config import REFERENCE

__all__ = ["Turn", "classify_role", "parse_turns"]

# A speaker label: a run of 1-5 ALL-CAPS tokens (allowing ``.'-``), an optional
# comma-delimited role clause, then a colon and whitespace. The leading boundary
# is validated in code (Python lookbehind must be fixed-width).
_LABEL = re.compile(
    r"(?P<name>[A-Z][A-Z.'\-]+(?:\s+[A-Z][A-Z.'&\-]+){0,4})"
    r"(?:,\s*(?P<role>[^:\n]{1,80}?))?"
    r":\s",
)

# Characters that legitimately precede a speaker label (sentence end / newline /
# start-of-text), used to reject mid-sentence ``WORD:`` false positives.
# includes curly close-quote/apostrophe (U+201D, U+2019) common in CNN text
_BOUNDARY_CHARS = frozenset(".?!:\"')]\u201d\u2019\n")

# (BEGIN/END VIDEO CLIP | VIDEOTAPE | AUDIO CLIP | AUDIOTAPE ...) markers that
# bracket pre-recorded material; a turn inside such a span is a played clip,
# not a live appearance.
_CLIP_MARKER = re.compile(r"\((BEGIN|END)\s+(?:VIDEO|AUDIO)[A-Z ]*\)")


@dataclass(frozen=True, slots=True)
class Turn:
    """One speaker turn with provenance offsets into the source text."""

    turn_index: int
    char_start: int  # start of the utterance (after the label) in source text
    char_end: int  # end of the utterance in source text
    speaker_raw: str
    role_raw: str | None
    name_norm: str  # canonical-within-transcript name (lower-cased), roster-resolved
    staff_flag: str  # "staff" | "guest" | "nonperson" | "unknown"
    source_mode: str  # "live" | "clip" -- played clip vs. live appearance
    utterance: str


def _clip_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges bracketed by ``(BEGIN ...)``/``(END ...)`` clip markers."""
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    for m in _CLIP_MARKER.finditer(text):
        if m.group(1) == "BEGIN":
            open_at = m.end()
        elif open_at is not None:  # END closes the current span
            spans.append((open_at, m.start()))
            open_at = None
    if open_at is not None:  # unclosed clip runs to end of text
        spans.append((open_at, len(text)))
    return spans


def _in_clip(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


@functools.lru_cache(maxsize=1)
def _role_lexicon() -> dict[str, set[str]]:
    """Load role keywords grouped by category (``staff``/``guest``)."""
    by_cat: dict[str, set[str]] = {"staff": set(), "guest": set()}
    path = REFERENCE / "role_lexicon.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            kw = (row.get("keyword") or "").strip().lower()
            cat = (row.get("category") or "").strip().lower()
            if kw and cat in by_cat:
                by_cat[cat].add(kw)
    return by_cat


@functools.lru_cache(maxsize=1)
def _nonperson_tokens() -> tuple[str, ...]:
    path = REFERENCE / "nonperson_speakers.txt"
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.lower())
    return tuple(out)


def _is_nonperson(name: str) -> bool:
    low = name.strip().lower()
    return any(low == tok or low.startswith(tok + " ") for tok in _nonperson_tokens())


def classify_role(role: str | None, name: str = "") -> str:
    """Classify a speaker as ``staff``/``guest``/``nonperson``/``unknown``.

    A role containing a staff keyword (host/anchor/correspondent/analyst/CNN…)
    is staff; any other non-empty role is treated as an external guest. With no
    role we cannot tell, so it is ``unknown`` (and usually resolved via roster).
    """
    if _is_nonperson(name):
        return "nonperson"
    if not role:
        return "unknown"
    low = role.lower()
    lex = _role_lexicon()
    if any(kw in low for kw in lex["staff"]):
        return "staff"
    return "guest"


def _surname(name_norm: str) -> str:
    return name_norm.split()[-1] if name_norm else name_norm


def _valid_boundary(text: str, start: int) -> bool:
    """True if ``start`` begins at a turn boundary (start/newline/sentence end)."""
    prefix = text[:start].rstrip(" \t")
    if not prefix:
        return True
    return prefix[-1] in _BOUNDARY_CHARS


def parse_turns(text: str, era_id: str | None = None) -> list[Turn]:
    """Split ``text`` into ordered :class:`Turn`s.

    ``era_id`` is accepted for forward-compatibility with era-specific tweaks;
    the boundary logic already handles both newline-delimited (old) and
    single-blob (modern) layouts.
    """
    if not text or not text.strip():
        return []

    # Pass 1: collect valid label matches (start offset, name, role).
    matches: list[tuple[int, int, str, str | None]] = []
    for m in _LABEL.finditer(text):
        if not _valid_boundary(text, m.start("name")):
            continue
        name = re.sub(r"\s+", " ", m.group("name")).strip()
        role = m.group("role")
        role = re.sub(r"\s+", " ", role).strip() if role else None
        matches.append((m.start("name"), m.end(), name, role))

    if not matches:
        return []

    # Pass 2: build the per-transcript roster from full-form labels (those with
    # a role, or multi-token names), surname -> (name_norm, staff_flag).
    roster: dict[str, tuple[str, str]] = {}
    for _start, _utt_start, name, role in matches:
        name_norm = name.lower()
        flag = classify_role(role, name)
        is_full = role is not None or len(name.split()) > 1
        if is_full and flag in {"staff", "guest"}:
            roster.setdefault(_surname(name_norm), (name_norm, flag))

    # Pass 3: emit turns, slicing utterances between consecutive labels.
    clip_spans = _clip_spans(text)
    turns: list[Turn] = []
    for i, (label_start, utt_start, name, role) in enumerate(matches):
        utt_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        utterance = text[utt_start:utt_end].strip()
        # recompute true offsets of the trimmed utterance within source text
        true_start = utt_start + (
            len(text[utt_start:utt_end]) - len(text[utt_start:utt_end].lstrip())
        )
        true_end = true_start + len(utterance)

        name_norm = name.lower()
        flag = classify_role(role, name)
        if flag == "unknown":  # bare continuation surname -> roster lookup
            hit = roster.get(_surname(name_norm))
            if hit:
                name_norm, flag = hit
        turns.append(
            Turn(
                turn_index=i,
                char_start=true_start,
                char_end=true_end,
                speaker_raw=name,
                role_raw=role,
                name_norm=name_norm,
                staff_flag=flag,
                source_mode="clip" if _in_clip(label_start, clip_spans) else "live",
                utterance=utterance,
            )
        )
    return turns
