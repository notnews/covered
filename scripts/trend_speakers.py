"""Full-corpus speaker-turn trends: concentration + named/attribute breakdowns.

Streams every era in chunks (memory-safe for the 1 GB+ files) and accumulates
external-guest turn counts keyed by
``(year, source_mode, canonical_id, is_president, office, party)``. "live" =
booked appearances; "clip" = played ``(BEGIN VIDEO CLIP)`` material; "all" =
both. From one pass it writes five annual tables to ``outputs/tables/``:

    python scripts/trend_speakers.py

* ``hhi_speakers_modes.csv``     -- HHI / top-k share / effective voices
* ``top_sources_annual.csv``     -- the top-N named sources per year + their share
* ``president_share_annual.csv`` -- sitting US president's share of turns
* ``office_share_annual.csv``    -- share by office bucket (incl. executive rollup)
* ``party_share_annual.csv``     -- share by R/D/I party tag
"""

from collections import defaultdict
from typing import cast

import pandas as pd

from covered import acquire, pipeline, schema
from covered.config import TABLES
from covered.hhi import concentration_metrics

TOP_N = 15  # named sources kept per (year, mode)
# Office buckets that make up the (US-leaning) executive branch rollup. "president"
# also catches foreign heads of state, so the clean US figure is president_share.
_EXECUTIVE = frozenset({"president", "vice_president", "cabinet_secretary", "press_secretary"})

# key: (year, mode, canonical_id, is_president, office, party) -> turn count
Key = tuple[int, str, str, bool, str, str]


def is_named(cid: object) -> bool:
    return bool(cid) and not str(cid).startswith("ambiguous:")


def _accumulate() -> dict[Key, int]:
    agg: dict[Key, int] = defaultdict(int)
    cols = ["year", "staff_flag", "source_mode", "canonical_id", "is_president", "office", "party"]
    for p in acquire.raw_files():
        n = 0
        for chunk in pd.read_csv(p, dtype=str, keep_default_na=False, chunksize=20000):
            chunk = schema.validate_csv(chunk)
            turns = pipeline.build_turns(chunk)
            if not turns.empty:
                sub = turns[turns["staff_flag"] == "guest"][cols].dropna(subset=["year"])
                grouped = sub.groupby(["year", "source_mode", *cols[3:]], dropna=False).size()
                for key, cnt in grouped.items():
                    y, mode, cid, isp, office, party = cast(
                        "tuple[int, str, str, bool, str, str | None]", key
                    )
                    if not is_named(cid):
                        continue
                    party_s = party if isinstance(party, str) and party else "none"
                    office_s = office if isinstance(office, str) and office else ""
                    c = int(cnt)
                    for m in (str(mode), "all"):
                        agg[(int(y), m, str(cid), bool(isp), office_s, party_s)] += c
            n += len(chunk)
            del chunk, turns
        print(f"[{p.name}] {n:,} segments done", flush=True)
    return agg


def main() -> None:
    agg = _accumulate()
    years = sorted({k[0] for k in agg})
    modes = ("all", "live", "clip")

    hhi_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    pres_rows: list[dict[str, object]] = []
    office_rows: list[dict[str, object]] = []
    party_rows: list[dict[str, object]] = []

    for mode in modes:
        for y in years:
            names: dict[str, int] = defaultdict(int)
            office_ct: dict[str, int] = defaultdict(int)
            party_ct: dict[str, int] = defaultdict(int)
            pres = 0
            for (yy, m, cid, isp, office, party), cnt in agg.items():
                if yy != y or m != mode:
                    continue
                names[cid] += cnt
                office_ct[office] += cnt
                party_ct[party] += cnt
                if isp:
                    pres += cnt
            total = sum(names.values())
            if not total:
                continue

            hhi_rows.append(
                {"year": y, "variant": f"external-{mode}", **concentration_metrics(names)}
            )
            pres_rows.append(
                {
                    "year": y,
                    "mode": mode,
                    "n_president_turns": pres,
                    "n_guest_turns": total,
                    "president_share": pres / total,
                }
            )
            for cid, cnt in sorted(names.items(), key=lambda kv: -kv[1])[:TOP_N]:
                top_rows.append(
                    {
                        "year": y,
                        "mode": mode,
                        "rank": 0,  # filled after sort below
                        "canonical_id": cid,
                        "n_turns": cnt,
                        "share": cnt / total,
                    }
                )
            exec_turns = sum(c for o, c in office_ct.items() if o in _EXECUTIVE)
            office_rows.append(
                {
                    "year": y,
                    "mode": mode,
                    "n_guest_turns": total,
                    "office_coverage": 1 - office_ct.get("", 0) / total,
                    "executive_share": exec_turns / total,
                    **{f"{o}_share": office_ct.get(o, 0) / total for o in sorted(office_ct) if o},
                }
            )
            party_rows.append(
                {
                    "year": y,
                    "mode": mode,
                    "n_guest_turns": total,
                    "R_share": party_ct.get("R", 0) / total,
                    "D_share": party_ct.get("D", 0) / total,
                    "I_share": party_ct.get("I", 0) / total,
                    "party_coverage": 1 - party_ct.get("none", 0) / total,
                }
            )

    # rank within each (mode, year) for the top-sources table
    top = pd.DataFrame(top_rows)
    top["rank"] = top.groupby(["mode", "year"])["n_turns"].rank(ascending=False, method="first")
    top["rank"] = top["rank"].astype(int)

    TABLES.mkdir(parents=True, exist_ok=True)
    _write(pd.DataFrame(hhi_rows), ["variant", "year"], "hhi_speakers_modes.csv")
    _write(top, ["mode", "year", "rank"], "top_sources_annual.csv")
    _write(pd.DataFrame(pres_rows), ["mode", "year"], "president_share_annual.csv")
    _write(pd.DataFrame(office_rows), ["mode", "year"], "office_share_annual.csv")
    _write(pd.DataFrame(party_rows), ["mode", "year"], "party_share_annual.csv")


def _write(df: pd.DataFrame, sort_by: list[str], name: str) -> None:
    out = df.sort_values(sort_by).reset_index(drop=True)
    dest = TABLES / name
    out.to_csv(dest, index=False)
    print(f"wrote {len(out)} rows -> {dest}", flush=True)


if __name__ == "__main__":
    main()
