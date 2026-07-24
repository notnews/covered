"""Measure (b): extract third-party quote attributions from body text.

A source is a PERSON or ORG credited with a statement, found two ways:

* **subject-of-cue-verb** (``Biden said``, ``"...", said Clinton``, ``the White
  House said``) — order-independent because it matches the dependency parse,
  not surface word order;
* **according-to** (``According to Obama, ...``).

Each attribution carries the source span's character offsets so it can be
audited against the source text. Self-references (the on-air speaker quoting
themselves) are excluded.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from covered.config import REFERENCE

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Span, Token

__all__ = ["Attribution", "extract_attributions"]

_SOURCE_TYPES = frozenset({"PERSON", "ORG"})


@dataclass(frozen=True, slots=True)
class Attribution:
    """One credited source with provenance offsets into the source text."""

    sentence_index: int
    char_start: int
    char_end: int
    source_span: str
    entity_type: str  # "PERSON" | "ORG"
    cue_verb: str  # cue lemma, or "according to"
    pattern_id: str  # "subj_cue" | "according_to"
    sentence_text: str


@functools.lru_cache(maxsize=1)
def _cue_lemmas() -> tuple[str, ...]:
    path = REFERENCE / "cue_verbs.txt"
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.lower())
    return tuple(out)


@functools.lru_cache(maxsize=2)
def _build_matcher(nlp: Language):  # type: ignore[no-untyped-def]
    """Build the DependencyMatcher with the subject-cue and according-to patterns."""
    from spacy.matcher import DependencyMatcher

    matcher = DependencyMatcher(nlp.vocab)
    matcher.add(
        "subj_cue",
        [
            [
                {
                    "RIGHT_ID": "cue",
                    "RIGHT_ATTRS": {
                        "POS": "VERB",
                        "LEMMA": {"IN": list(_cue_lemmas())},
                    },
                },
                {
                    "LEFT_ID": "cue",
                    "REL_OP": ">",
                    "RIGHT_ID": "subj",
                    "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}},
                },
            ]
        ],
    )
    matcher.add(
        "according_to",
        [
            [
                {
                    "RIGHT_ID": "accord",
                    "RIGHT_ATTRS": {"LEMMA": "accord", "DEP": "prep"},
                },
                {
                    "LEFT_ID": "accord",
                    "REL_OP": ">",
                    "RIGHT_ID": "to",
                    "RIGHT_ATTRS": {"LOWER": "to"},
                },
                {
                    "LEFT_ID": "to",
                    "REL_OP": ">",
                    "RIGHT_ID": "src",
                    "RIGHT_ATTRS": {"DEP": "pobj"},
                },
            ]
        ],
    )
    return matcher


def _entity_for_token(token: Token) -> Span | None:
    """Return the named entity span containing ``token`` (PERSON/ORG), else None."""
    ent = token.ent_type_
    if ent not in _SOURCE_TYPES:
        return None
    for span in token.doc.ents:
        if span.start <= token.i < span.end and span.label_ in _SOURCE_TYPES:
            return span
    return None


def _normalize(name: str) -> str:
    return " ".join(name.lower().split())


def extract_attributions(
    text: str,
    nlp: Language,
    exclude_names: frozenset[str] | set[str] = frozenset(),
) -> list[Attribution]:
    """Extract credited sources from ``text``.

    ``exclude_names`` (lower-cased) drops self-references — typically the on-air
    speaker of the utterance being parsed.
    """
    if not text or not text.strip():
        return []
    doc: Doc = nlp(text)
    sent_index = {sent.start: i for i, sent in enumerate(doc.sents)}

    def sentence_of(token: Token) -> tuple[int, str]:
        sent = token.sent
        return sent_index.get(sent.start, 0), sent.text

    exclude = {_normalize(n) for n in exclude_names}
    matcher = _build_matcher(nlp)
    seen: set[tuple[int, int]] = set()
    out: list[Attribution] = []

    for match_id, token_ids in matcher(doc):
        pattern_id = nlp.vocab.strings[match_id]
        if pattern_id == "subj_cue":
            cue_tok, src_tok = doc[token_ids[0]], doc[token_ids[1]]
            cue_verb = cue_tok.lemma_.lower()
        else:  # according_to: token_ids = [accord, to, src]
            src_tok = doc[token_ids[-1]]
            cue_verb = "according to"

        span = _entity_for_token(src_tok)
        if span is None:
            continue
        key = (span.start_char, span.end_char)
        if key in seen:
            continue
        if _normalize(span.text) in exclude:
            continue
        seen.add(key)
        s_idx, s_text = sentence_of(src_tok)
        out.append(
            Attribution(
                sentence_index=s_idx,
                char_start=span.start_char,
                char_end=span.end_char,
                source_span=span.text,
                entity_type=span.label_,
                cue_verb=cue_verb,
                pattern_id=pattern_id,
                sentence_text=s_text,
            )
        )

    out.sort(key=lambda a: a.char_start)
    return out
