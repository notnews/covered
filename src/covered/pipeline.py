"""Orchestration: turn raw transcript frames into extraction tables and HHI series.

Every emitted row carries provenance (uid/url/show_code/host/dateline/headline/
subhead/offsets) so any HHI count can be traced back to its source segment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

from covered import attribution, entities, roles, speakers
from covered.hhi import concentration_metrics
from covered.provenance import load_show_map, parse_provenance

if TYPE_CHECKING:
    from spacy.language import Language

__all__ = [
    "annual_hhi_attributions",
    "annual_hhi_speakers",
    "build_attributions",
    "build_turns",
]

# columns copied from Provenance onto every extracted row
_PROV_COLS = (
    "uid",
    "url",
    "path",
    "channel_name",
    "program_name",
    "headline",
    "subhead",
    "show_code",
    "host",
    "air_date",
    "time",
    "timezone",
    "era_id",
)


def _prov_dict(row: pd.Series, show_map: dict[str, str]) -> dict[str, object]:
    p = parse_provenance(
        {str(k): v for k, v in row.to_dict().items()}, show_map=show_map
    )
    d: dict[str, object] = {c: getattr(p, c) for c in _PROV_COLS}
    d["year"] = p.air_date.year if p.air_date else None
    return d


def build_turns(
    df: pd.DataFrame, show_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Measure (a): explode each segment into speaker turns with provenance."""
    show_map = load_show_map() if show_map is None else show_map
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        prov = _prov_dict(row, show_map)
        for t in speakers.parse_turns(
            str(row.get("text", "") or ""), era_id=cast("str | None", prov["era_id"])
        ):
            records.append(
                {
                    **prov,
                    "turn_index": t.turn_index,
                    "char_start": t.char_start,
                    "char_end": t.char_end,
                    "speaker_raw": t.speaker_raw,
                    "role_raw": t.role_raw,
                    "name_norm": t.name_norm,
                    "staff_flag": t.staff_flag,
                    "source_mode": t.source_mode,
                }
            )
    turns = pd.DataFrame.from_records(records)
    if not turns.empty:
        turns["canonical_id"] = entities.resolve_mentions(turns["name_norm"].tolist())
        # Tier-1 deterministic enrichment (office/party from the role text, and
        # whether the speaker is the US president sitting on the air date). The
        # role parsers tolerate pandas NaN (a non-str) and return the empty result.
        turns["office"] = turns["role_raw"].map(roles.classify_office)
        turns["party"] = turns["role_raw"].map(roles.parse_party)
        turns["is_president"] = [
            roles.is_sitting_president(cid, air)
            for cid, air in zip(turns["canonical_id"], turns["air_date"], strict=True)
        ]
    return turns


def build_attributions(
    df: pd.DataFrame,
    nlp: Language,
    show_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Measure (b): extract cited sources per segment with provenance.

    On-air speakers of the segment are excluded as self-references.
    """
    show_map = load_show_map() if show_map is None else show_map
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        prov = _prov_dict(row, show_map)
        text = str(row.get("text", "") or "")
        on_air = {
            t.name_norm
            for t in speakers.parse_turns(
                text, era_id=cast("str | None", prov["era_id"])
            )
        }
        for a in attribution.extract_attributions(text, nlp, exclude_names=on_air):
            records.append(
                {
                    **prov,
                    "sentence_index": a.sentence_index,
                    "char_start": a.char_start,
                    "char_end": a.char_end,
                    "source_span": a.source_span,
                    "entity_type": a.entity_type,
                    "cue_verb": a.cue_verb,
                    "pattern_id": a.pattern_id,
                    "sentence_text": a.sentence_text,
                }
            )
    atts = pd.DataFrame.from_records(records)
    if not atts.empty:
        atts["canonical_id"] = entities.resolve_mentions(atts["source_span"].tolist())
    return atts


def _is_named(canonical_id: object) -> bool:
    cid = str(canonical_id)
    return bool(cid) and not cid.startswith("ambiguous:")


def _annual(
    df: pd.DataFrame,
    measure: str,
    variant: str,
) -> pd.DataFrame:
    """Compute per-year concentration metrics from a (year, canonical_id) frame."""
    rows: list[dict[str, object]] = []
    for year, grp in df.groupby("year"):
        counts = {
            str(k): float(v)
            for k, v in grp["canonical_id"].value_counts().to_dict().items()
        }
        metrics = concentration_metrics(counts)
        rows.append(
            {
                "year": int(cast("int", year)),
                "measure": measure,
                "variant": variant,
                **metrics,
            }
        )
    if not rows:
        cols = ["year", "measure", "variant", *concentration_metrics({}).keys()]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def annual_hhi_speakers(
    turns: pd.DataFrame,
    variant: str = "external",
    mode: str = "all",
) -> pd.DataFrame:
    """Annual speaker-turn HHI.

    ``variant='external'`` drops CNN staff; ``'all'`` keeps staff and guests
    (always excluding non-person and unresolved names). ``mode`` restricts to
    played clips (``'clip'``), live appearances (``'live'``), or both
    (``'all'``) -- i.e. whose words CNN *plays* vs. whom it *books* live.
    """
    keep = {"guest"} if variant == "external" else {"guest", "staff"}
    df = turns[turns["staff_flag"].isin(keep) & turns["canonical_id"].map(_is_named)]
    if mode != "all":
        df = df[df["source_mode"] == mode]
    df = df[df["year"].notna()]
    label = variant if mode == "all" else f"{variant}-{mode}"
    return _annual(df, measure="speakers", variant=label)


def annual_hhi_attributions(
    atts: pd.DataFrame,
    entity_type: str = "PERSON",
) -> pd.DataFrame:
    """Annual cited-source HHI, deduped once per (source, segment)."""
    df = atts[
        (atts["entity_type"] == entity_type) & atts["canonical_id"].map(_is_named)
    ]
    # dedup once per (source, segment); url is the always-unique segment key
    df = df[df["year"].notna()].drop_duplicates(["canonical_id", "url", "year"])
    variant = entity_type.lower()
    return _annual(df, measure="attributions", variant=variant)
