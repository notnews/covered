"""Assign each transcript to a corpus epoch based on its air date.

The three epochs correspond to distinct CNN HTML/text formats; the speaker and
attribution parsers branch on the era id to stay robust to format drift.
"""

from __future__ import annotations

from datetime import date

from covered.config import CORPUS_START, ERA_IDS, ERA_STARTS

__all__ = ["ERA_IDS", "era_for_date"]


def era_for_date(day: date) -> str:
    """Return the era id (``era1``/``era2``/``era3``) for an air date.

    Raises ``ValueError`` for dates before the corpus start (2000-01-01).
    """
    if day < CORPUS_START:
        raise ValueError(f"{day.isoformat()} is before the corpus start {CORPUS_START.isoformat()}")
    era = ERA_STARTS[0][0]
    for era_id, start in ERA_STARTS:
        if day >= start:
            era = era_id
    return era
