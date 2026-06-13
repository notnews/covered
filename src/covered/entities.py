"""Entity resolution: map surface name forms to canonical individuals.

Strategy is deterministic-first:

1. **Normalize** — strip honorifics, party/affiliation tags, punctuation; the
   comma in a CNN label separates name from role, so only the pre-comma part is
   the name.
2. **Alias overrides** — a curated table resolves the high-frequency head
   (presidents, anchors), which dominates the HHI.
3. **Block + disambiguate** — group remaining mentions by a phonetic key on the
   surname; a bare surname resolves to the unique full name in its block, or is
   marked ``ambiguous:<surname>`` when several full names collide (we never
   silently over-merge).
"""

from __future__ import annotations

import csv
import functools
import re
from collections.abc import Mapping, Sequence

from metaphone import doublemetaphone

from covered.config import REFERENCE

__all__ = [
    "block_key",
    "load_aliases",
    "normalize_name",
    "resolve_mentions",
]

_PARENS = re.compile(r"\([^)]*\)")
_DROP_PUNCT = re.compile(r"[.,;:\"]+")
_WS = re.compile(r"\s+")


@functools.lru_cache(maxsize=1)
def _titles() -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(single_word_titles, multiword_titles)`` normalized for matching."""
    single: set[str] = set()
    multi: set[str] = set()
    for line in (REFERENCE / "titles.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip().lower().rstrip(".")
        if not line or line.startswith("#"):
            continue
        (multi if " " in line else single).add(line)
    return frozenset(single), frozenset(multi)


def _strip_titles(tokens: list[str]) -> list[str]:
    single, multi = _titles()
    out = list(tokens)
    while out:
        if len(out) >= 2 and f"{out[0]} {out[1]}" in multi:
            out = out[2:]
        elif out[0] in single and len(out) > 1:
            out = out[1:]
        else:
            break
    return out


def normalize_name(raw: str) -> str:
    """Normalize a raw name to a canonical surface form (lower-cased, de-titled).

    Cuts at the first comma (name/role separator in CNN labels), removes
    parenthetical party/affiliation tags, drops punctuation except internal
    apostrophes and hyphens, and strips leading honorifics.
    """
    if not raw:
        return ""
    s = raw.lower().split(",")[0]
    s = _PARENS.sub(" ", s)
    s = _DROP_PUNCT.sub(" ", s)
    tokens = _WS.sub(" ", s).strip().split()
    tokens = _strip_titles(tokens)
    return " ".join(tokens)


def _surname(norm: str) -> str:
    return norm.split()[-1] if norm else ""


def block_key(norm: str) -> str:
    """Phonetic blocking key on the surname (absorbs rush-transcript misspellings)."""
    surname = _surname(norm)
    if not surname:
        return ""
    primary = doublemetaphone(surname)[0]
    return primary or surname


@functools.lru_cache(maxsize=1)
def load_aliases() -> dict[str, str]:
    """Load curated ``normalized-surface -> canonical_id`` overrides."""
    mapping: dict[str, str] = {}
    lines = (REFERENCE / "aliases.csv").read_text(encoding="utf-8").splitlines()
    rows = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    for row in csv.DictReader(rows):
        surface = normalize_name(row.get("surface", ""))
        canonical = (row.get("canonical_id") or "").strip().lower()
        if surface and canonical:
            mapping[surface] = canonical
    return mapping


def resolve_mentions(
    raw_names: Sequence[str],
    alias_map: Mapping[str, str] | None = None,
) -> list[str]:
    """Resolve a sequence of raw name mentions to canonical ids (order preserved).

    Bare surnames resolve to the unique full name sharing their phonetic block,
    or ``ambiguous:<surname>`` if several full names collide, or to themselves
    when no full name is present.
    """
    aliases = load_aliases() if alias_map is None else alias_map
    norms = [normalize_name(r) for r in raw_names]

    # Full-name candidates per block (multi-token mentions, not alias-handled).
    full_by_block: dict[str, set[str]] = {}
    for norm in norms:
        if norm and norm not in aliases and len(norm.split()) > 1:
            full_by_block.setdefault(block_key(norm), set()).add(norm)

    out: list[str] = []
    for norm in norms:
        if not norm:
            out.append("")
            continue
        if norm in aliases:
            out.append(aliases[norm])
            continue
        if len(norm.split()) > 1:  # full name resolves to itself
            out.append(norm)
            continue
        # bare surname: look at full candidates sharing the block and surname
        candidates = {
            full for full in full_by_block.get(block_key(norm), set()) if _surname(full) == norm
        }
        if len(candidates) == 1:
            out.append(next(iter(candidates)))
        elif len(candidates) > 1:
            out.append(f"ambiguous:{norm}")
        else:
            out.append(norm)
    return out
