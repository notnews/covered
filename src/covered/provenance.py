"""Parse per-segment provenance from CSV metadata.

Every extracted record (speaker turn or attribution) is stamped with a
:class:`Provenance` so any HHI count can be traced back to its source segment
and audited. The slug (``uid``/``url``) is decomposed into a show code and
segment number; the human show name comes from ``program.name`` and an optional
host is looked up from a reference show map.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from covered.config import REFERENCE
from covered.eras import era_for_date

__all__ = [
    "Provenance",
    "load_show_map",
    "parse_provenance",
    "parse_uid",
    "parse_url_date",
]

_URL_DATE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Audit trail attached to every extracted record."""

    uid: str
    url: str
    path: str
    channel_name: str
    program_name: str
    headline: str  # == program.name (cnnTransStoryHead); named for clarity
    subhead: str
    show_code: str | None
    segment_index: int | None
    host: str | None
    air_date: date | None
    url_date: date | None
    time: str
    timezone: str
    era_id: str | None


def load_show_map(path: Path | None = None) -> dict[str, str]:
    """Load the curated ``show_code -> host`` reference table.

    Codes are lower-cased. Rows without a host are skipped (the host column is
    the value-add; the show name itself comes from ``program.name``).
    """
    path = path or (REFERENCE / "show_map.csv")
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("show_code") or "").strip().lower()
            host = (row.get("host") or "").strip()
            if code and host:
                mapping[code] = host
    return mapping


def parse_uid(uid: str) -> tuple[str | None, int | None]:
    """Split a uid like ``acd.01`` into ``(show_code, segment_index)``.

    The show code is the leading dotted token; the segment is the trailing
    token when it is purely numeric, else ``None``. Returns ``(None, None)`` for
    an empty uid.
    """
    uid = (uid or "").strip().lower()
    if not uid:
        return None, None
    tokens = uid.split(".")
    show_code = tokens[0] or None
    segment_index: int | None = None
    if len(tokens) > 1 and tokens[-1].isdigit():
        segment_index = int(tokens[-1])
    return show_code, segment_index


def parse_url_date(url: str) -> date | None:
    """Extract a ``YYYY.MM.DD`` air date embedded in a transcript URL/path."""
    m = _URL_DATE.search(url or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _air_date_from_columns(row: Mapping[str, object]) -> date | None:
    try:
        return date(
            int(str(row["year"])), int(str(row["month"])), int(str(row["date"]))
        )
    except (KeyError, TypeError, ValueError):
        return None


def parse_provenance(
    row: Mapping[str, object],
    show_map: Mapping[str, str] | None = None,
) -> Provenance:
    """Build a :class:`Provenance` from one CSV row.

    ``show_map`` maps show codes (e.g. ``acd``) to a host/program label; missing
    codes yield ``host=None``.
    """
    show_map = show_map or {}
    uid = str(row.get("uid", "") or "")
    url = str(row.get("url", "") or "")
    if not uid and url:  # older eras leave uid/path blank; the slug is in the URL
        last = url.rstrip("/").split("/")[-1]
        uid = last[:-5] if last.endswith(".html") else last
    show_code, segment_index = parse_uid(uid)
    air_date = _air_date_from_columns(row)
    program_name = str(row.get("program.name", "") or "")
    return Provenance(
        uid=uid,
        url=url,
        path=str(row.get("path", "") or ""),
        channel_name=str(row.get("channel.name", "") or ""),
        program_name=program_name,
        headline=program_name,
        subhead=str(row.get("subhead", "") or ""),
        show_code=show_code,
        segment_index=segment_index,
        host=show_map.get(show_code) if show_code else None,
        air_date=air_date,
        url_date=parse_url_date(url),
        time=str(row.get("time", "") or ""),
        timezone=str(row.get("timezone", "") or ""),
        era_id=era_for_date(air_date) if air_date else None,
    )
