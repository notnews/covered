"""Full-corpus frequency index of raw speaker-role strings.

The role clause in a ``NAME, ROLE:`` label is the only in-text signal of who a
speaker *is* -- ``NYU LAW PROFESSOR``, ``SENIOR FELLOW, ATLANTIC COUNCIL``,
``UKRAINIAN REFUGEE``. Classifying those strings is what the source taxonomy is
built on, so this script counts them across the whole corpus and writes the
ranked list that hand-coding works down:

    python scripts/build_role_index.py                 # full corpus
    python scripts/build_role_index.py --sample 2000   # 2000 segments per era

Writes ``data/interim/role_strings.csv``:

    role_raw, staff_flag, n_turns, n_docs, first_year, last_year,
    example_speaker, share, cum_share

``cum_share`` is over guest turns only and is the number that matters for
coding effort: it says what fraction of external-source turns are covered once
you have coded down to that row.

Also prints a coverage diagnostic -- what share of guest turns the current
office lexicon classifies, split by whether the role was labelled on the turn
itself or inherited from the transcript roster. That is the gap the taxonomy
has to close.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from covered import acquire, roles, schema, speakers
from covered.config import INTERIM
from covered.provenance import parse_provenance

Key = tuple[str, str]  # (role_raw, staff_flag)


@dataclass
class RoleStat:
    """Running aggregate for one ``(role_raw, staff_flag)`` pair."""

    n_turns: int = 0
    first_year: int | None = None
    last_year: int | None = None
    example_speaker: str = ""
    docs: set[str] = field(default_factory=set)

    def observe(self, year: int | None, speaker: str, uid: str) -> None:
        self.n_turns += 1
        if not self.example_speaker:
            self.example_speaker = speaker
        if year is not None:
            fy, ly = self.first_year, self.last_year
            self.first_year = year if fy is None else min(fy, year)
            self.last_year = year if ly is None else max(ly, year)
        if uid:
            self.docs.add(uid)


def _accumulate(sample: int | None) -> tuple[dict[Key, RoleStat], dict[str, int]]:
    agg: dict[Key, RoleStat] = defaultdict(RoleStat)
    diag: dict[str, int] = defaultdict(int)  # coverage counters, over guest turns

    for p in acquire.raw_files():
        n = 0
        for chunk in pd.read_csv(p, dtype=str, keep_default_na=False, chunksize=20000):
            if sample is not None:
                if n >= sample:
                    break
                chunk = chunk.head(sample - n)  # honour --sample within a chunk
            chunk = schema.validate_csv(chunk)
            for _, row in chunk.iterrows():
                prov = parse_provenance(row)
                uid = prov.uid or prov.url
                year = prov.air_date.year if prov.air_date else None
                for t in speakers.parse_turns(
                    str(row.get("text", "") or ""), prov.era_id
                ):
                    if t.staff_flag == "guest":
                        diag["guest_turns"] += 1
                        diag[f"role_source_{t.role_source}"] += 1
                        if roles.classify_office(t.role_raw):
                            diag["office_hit"] += 1
                            diag[f"office_hit_{t.role_source}"] += 1
                    if t.role_raw:
                        agg[(t.role_raw, t.staff_flag)].observe(
                            year, t.speaker_raw, uid
                        )
            n += len(chunk)
            del chunk
        print(f"[{p.name}] {n:,} segments done", flush=True)
    return agg, diag


def _report_coverage(diag: dict[str, int]) -> None:
    g = diag.get("guest_turns", 0)
    if not g:
        print("no guest turns seen")
        return
    local = diag.get("role_source_local", 0)
    roster = diag.get("role_source_roster", 0)
    hit = diag.get("office_hit", 0)
    hit_local = diag.get("office_hit_local", 0)
    print("\ncoverage over guest turns")
    print(f"  guest turns                    {g:>12,}")
    print(f"  role labelled on the turn      {local:>12,}  {local / g:>7.3f}")
    print(f"  role inherited from roster     {roster:>12,}  {roster / g:>7.3f}")
    print(f"  no role anywhere               {g - local - roster:>12,}")
    print("\n  classified by the office lexicon")
    print(f"    labelled turns only (before) {hit_local:>12,}  {hit_local / g:>7.3f}")
    print(f"    incl. inherited     (after)  {hit:>12,}  {hit / g:>7.3f}")
    print(f"    unclassified                 {g - hit:>12,}  {1 - hit / g:>7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sample", type=int, default=None, help="segments per era file (default: all)"
    )
    args = ap.parse_args()

    agg, diag = _accumulate(args.sample)
    df = (
        pd.DataFrame(
            [
                {
                    "role_raw": role,
                    "staff_flag": flag,
                    "n_turns": s.n_turns,
                    "n_docs": len(s.docs),
                    "first_year": s.first_year,
                    "last_year": s.last_year,
                    "example_speaker": s.example_speaker,
                }
                for (role, flag), s in agg.items()
            ]
        )
        .sort_values("n_turns", ascending=False)
        .reset_index(drop=True)
    )

    # Shares are over guest turns: staff roles are the network's own people,
    # classified by citation mode rather than by the taxonomy's sector axis.
    is_guest = df["staff_flag"] == "guest"
    guest_total = int(df.loc[is_guest, "n_turns"].sum())
    df["share"] = df["n_turns"].where(is_guest) / guest_total
    df["cum_share"] = (df["n_turns"] * is_guest).cumsum().where(is_guest) / guest_total

    INTERIM.mkdir(parents=True, exist_ok=True)
    dest = INTERIM / "role_strings.csv"
    df.to_csv(dest, index=False)

    print(f"\nwrote {len(df):,} distinct role strings -> {dest}")
    print(f"  distinct guest role strings:  {int(is_guest.sum()):,}")
    print(f"  guest turns with a role:      {guest_total:,}")
    for target in (0.50, 0.80, 0.90):
        need = int((df.loc[is_guest, "cum_share"] < target).sum()) + 1
        print(f"  guest roles to hand-code for {target:.0%} of them: {need:,}")
    _report_coverage(diag)


if __name__ == "__main__":
    main()
