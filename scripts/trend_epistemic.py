"""Annual series of the evidentiary standing of cable-news sources.

The question is the one in sourcerer's motivating essay: rather than checking
claims one at a time, assess the residue of the process that produced them. This
script measures the third of the essay's three tests -- the credibility of the
sources -- as the distribution of source turns over an ordinal scale of
*evidentiary proximity* to the events being reported:

    direct         witnessed or took part      principal, participant,
                                               eyewitness, personal_experience
    institutional  speaks for a body with      spokesperson
                   access
    expert         general knowledge, no       expert
                   access to this event
    commentary     opinion, no special access  commentator

    python scripts/trend_epistemic.py              # full corpus
    python scripts/trend_epistemic.py --every 12   # systematic 1-in-12 sample

Sampling is systematic (every Nth segment), never head-of-file: the era files
are ordered by air date, so a head sample covers only the opening weeks of each
era and yields a "trend" made of a handful of dates.

Writes ``outputs/tables/epistemic_annual.csv`` (year x mode x staff_flag x
proximity x how the label was decided) and prints the headline direct:commentary
ratio by year.

The headline is reported under three coding regimes because a large share of
epistemic labels are currently imputed from sector rather than read off the role
string. If the trend only survives under imputation it is an artefact of the
sector defaults, not a fact about the corpus:

    lexicon   only turns whose epistemic role came from the epistemic lexicon
    imputed   plus turns given their sector's default role
    all       plus turns resolved by the standing rule and name honorifics
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd

from covered import acquire, schema, speakers, taxonomy
from covered.config import TABLES
from covered.provenance import parse_provenance

# Ordinal scale: how close the source stands to the events being reported.
PROXIMITY: dict[str, str] = {
    "principal": "direct",
    "participant": "direct",
    "eyewitness": "direct",
    "personal_experience": "direct",
    "spokesperson": "institutional",
    "expert": "expert",
    "commentator": "commentary",
    "popular_opinion": "commentary",
    "unresolved": "unresolved",
}

# Which epistemic_source values each reporting regime admits.
REGIMES: dict[str, frozenset[str]] = {
    "lexicon": frozenset({"lexicon"}),
    "imputed": frozenset({"lexicon", "sector_default"}),
    "all": frozenset(
        {"lexicon", "sector_default", "standing_rule", "name_title", "topic_needed"}
    ),
}

Key = tuple[int, str, str, str, str]  # year, mode, staff_flag, proximity, epi_source


def _accumulate(every: int) -> dict[Key, int]:
    """Aggregate turn counts, keeping every ``every``-th segment.

    Systematic rather than head-of-file sampling: the era files are ordered by
    air date, so taking the first N rows of each would sample only the opening
    weeks of each era and produce a "trend" made of a handful of dates.
    """
    agg: dict[Key, int] = defaultdict(int)
    for p in acquire.raw_files():
        seen = kept = 0
        for chunk in pd.read_csv(p, dtype=str, keep_default_na=False, chunksize=20000):
            seen += len(chunk)
            if every > 1:
                chunk = chunk.iloc[list(range(0, len(chunk), every))]
            kept += len(chunk)
            chunk = schema.validate_csv(chunk)
            for _, row in chunk.iterrows():
                prov = parse_provenance(row)
                if prov.air_date is None:
                    continue
                year = prov.air_date.year
                for t in speakers.parse_turns(
                    str(row.get("text", "") or ""), prov.era_id
                ):
                    if t.staff_flag not in {"guest", "staff"}:
                        continue
                    label = taxonomy.classify(t.role_raw, t.speaker_raw)
                    agg[
                        (
                            year,
                            t.source_mode,
                            t.staff_flag,
                            PROXIMITY.get(label.epistemic_role, "unresolved"),
                            label.epistemic_source,
                        )
                    ] += 1
            del chunk
        print(f"[{p.name}] {kept:,} of {seen:,} segments", flush=True)
    return agg


def _ratio_series(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Direct-vs-commentary shares by year for one coding regime."""
    keep = df[df["epistemic_source"].isin(sorted(REGIMES[regime]))]
    wide = (
        keep.pivot_table(
            index="year", columns="proximity", values="n_turns", aggfunc="sum"
        )
        .fillna(0)
        .drop(columns=["unresolved"], errors="ignore")
    )
    total = wide.sum(axis=1)
    out = pd.DataFrame({"year": wide.index, "n_turns": total.to_numpy()})
    for col in ("direct", "institutional", "expert", "commentary"):
        out[f"{col}_share"] = (
            wide[col].to_numpy() / total.to_numpy()
            if col in wide
            else 0.0  # regime may admit no turns of this proximity
        )
    # The headline: how many direct-access turns per commentary turn.
    out["direct_per_commentary"] = out["direct_share"] / out[
        "commentary_share"
    ].replace(0, pd.NA)
    out["regime"] = regime
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--every",
        type=int,
        default=1,
        help="keep every Nth segment (default 1 = full corpus)",
    )
    ap.add_argument(
        "--staff",
        choices=["guest", "staff", "both"],
        default="guest",
        help="whose turns to count (default: external guests)",
    )
    args = ap.parse_args()

    agg = _accumulate(args.every)
    df = pd.DataFrame(
        [
            {
                "year": y,
                "mode": m,
                "staff_flag": s,
                "proximity": p,
                "epistemic_source": e,
                "n_turns": c,
            }
            for (y, m, s, p, e), c in agg.items()
        ]
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    dest = TABLES / "epistemic_annual.csv"
    df.sort_values(["year", "mode", "staff_flag", "proximity"]).to_csv(
        dest, index=False
    )
    print(f"\nwrote {len(df):,} rows -> {dest}")

    sub = df if args.staff == "both" else df.loc[df["staff_flag"] == args.staff]
    # 2025 is Jan-Mar only; excluded from the trend, kept in the written table.
    sub = sub[sub["year"] < 2025]

    print(f"\n=== direct-access vs commentary, {args.staff} turns ===")
    for regime in ("lexicon", "imputed", "all"):
        s = _ratio_series(sub, regime)
        if s.empty:
            continue
        first, last = s.iloc[0], s.iloc[-1]
        print(
            f"\n{regime:>8}:  {int(first.year)} direct {first.direct_share:.3f} / "
            f"commentary {first.commentary_share:.3f}"
            f"   ->   {int(last.year)} direct {last.direct_share:.3f} / "
            f"commentary {last.commentary_share:.3f}"
        )
        print(f"          {'year':>6} {'n':>9} {'direct':>8} {'comm':>8} {'d/c':>7}")
        for row in s.to_dict("records"):
            dpc = row["direct_per_commentary"]
            ratio = "" if pd.isna(dpc) else f"{dpc:7.2f}"
            print(
                f"          {int(row['year']):>6} {int(row['n_turns']):>9,} "
                f"{row['direct_share']:>8.3f} {row['commentary_share']:>8.3f} "
                f"{ratio:>7}"
            )


if __name__ == "__main__":
    main()
