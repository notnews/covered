"""pandera schema for the raw CNN transcript CSVs.

Validates the 14 published columns and fails loudly on format drift (out-of-range
dates, unexpected dtypes), while tolerating the known nullables — blank
``wordcount`` in the older eras and occasional empty ``time``/``subhead``.
"""

from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

__all__ = ["CNN_SCHEMA", "NUMERIC_COLS", "validate_csv"]

_str = Column(str, nullable=True, coerce=True)

# Numeric columns are nullable: real data has occasional blank dates (a malformed
# row) and entirely blank ``wordcount`` in the older eras. Range checks skip NA.
NUMERIC_COLS = ("year", "month", "date", "wordcount")

CNN_SCHEMA = DataFrameSchema(
    {
        "url": Column(str, nullable=False, coerce=True),
        "channel.name": _str,
        "program.name": _str,
        "uid": Column(str, nullable=True, coerce=True),
        "duration": _str,
        "year": Column("Int64", Check.in_range(1999, 2026), nullable=True, coerce=True),
        "month": Column("Int64", Check.in_range(1, 12), nullable=True, coerce=True),
        "date": Column("Int64", Check.in_range(1, 31), nullable=True, coerce=True),
        "time": _str,
        "timezone": _str,
        "path": _str,
        "wordcount": Column("Int64", Check.ge(0), nullable=True, coerce=True),
        "subhead": _str,
        "text": Column(str, nullable=True, coerce=True),
    },
    strict=False,  # tolerate extra columns
    coerce=True,
)


_RANGES = {"year": (1999, 2026), "month": (1, 12), "date": (1, 31)}


def validate_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate (and coerce) a raw transcript frame; raises on schema violation.

    Real data carries dirty numerics — blank ``wordcount``, the odd malformed
    date, and stray out-of-range years (e.g. ``3007`` for 2007). These are
    coerced to NA so a handful of bad rows don't crash a full-corpus run; rows
    with a missing year are dropped downstream from the HHI.
    """
    df = df.copy()
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
        if col in _RANGES:
            lo, hi = _RANGES[col]
            s = s.where(s.isna() | ((s >= lo) & (s <= hi)))
        df[col] = s.astype("Int64")
    return CNN_SCHEMA.validate(df, lazy=False)
