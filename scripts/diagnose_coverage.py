"""Characterise what the uncoded role strings are, and which way they bias.

Worst-case bounds treat the uncoded turns as capable of being anything, which is
true and useless: they come out wider than the signal. This asks the answerable
version instead -- is the uncoded residual plausibly *random* with respect to
the taxonomy's axes, and if not, in which direction does it push?

The test exploits the fact that codability is strongly related to how often a
role string recurs: standardised titles ("FORMER FBI DIRECTOR") repeat and match
keywords, while one-off descriptions ("HALEIGH'S BABY-SITTER") do neither. So
among the strings we *can* code, compare composition across frequency
quintiles. If composition shifts with rarity, then the uncoded mass -- which is
disproportionately rare -- is not a random sample of the corpus, and the shift's
direction estimates the sign of the bias.

    python scripts/diagnose_coverage.py

Reads ``data/interim/role_strings.csv`` (build it with build_role_index.py) and
writes ``outputs/tables/coverage_diagnosis.csv``.
"""

from __future__ import annotations

import pandas as pd

from covered import taxonomy
from covered.config import INTERIM, TABLES

QUINTILES = ["rarest", "2", "3", "4", "commonest"]


def main() -> None:
    df = pd.read_csv(INTERIM / "role_strings.csv")
    g = df[df["staff_flag"] == "guest"].copy()
    labels = [taxonomy.classify(str(r)) for r in g["role_raw"]]
    g["sector"] = [x.sector for x in labels]
    g["epistemic_role"] = [x.epistemic_role for x in labels]

    coded = g[g["sector"] != "unknown"]
    uncoded = g[g["sector"] == "unknown"]
    total = int(g["n_turns"].sum())

    print(f"guest turns {total:,}")
    print(f"  coded    {int(coded['n_turns'].sum()):>12,}  {len(coded):>8,} strings")
    print(
        f"  uncoded  {int(uncoded['n_turns'].sum()):>12,}  {len(uncoded):>8,} strings"
    )
    print("\nis the uncoded residual made of rarer strings?")
    print(f"  coded    median {coded['n_turns'].median():.0f} turns per string")
    print(f"  uncoded  median {uncoded['n_turns'].median():.0f} turns per string")

    quint = pd.qcut(coded["n_turns"].rank(method="first"), 5, labels=QUINTILES)
    rows: list[dict[str, object]] = []
    for axis in ("sector", "epistemic_role"):
        share = (
            coded.groupby([quint, axis], observed=True)["n_turns"]
            .sum()
            .unstack(fill_value=0)
        )
        share = share.div(share.sum(axis=1), axis=0)
        gap = (share.loc["rarest"] - share.loc["commonest"]).sort_values()
        print(f"\n{axis}: rarest minus commonest quintile, largest gaps")
        for key in list(gap.index[:4]) + list(gap.index[-4:]):
            print(f"  {gap[key] * 100:+6.1f} pp  {key}")
        for q in QUINTILES:
            for key, val in share.loc[q].items():
                rows.append(
                    {"axis": axis, "quintile": q, "category": key, "share": val}
                )

    TABLES.mkdir(parents=True, exist_ok=True)
    dest = TABLES / "coverage_diagnosis.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print(
        "\nRead the gaps as the sign of the bias: categories over-represented among\n"
        "rare strings are under-counted in the headline, because rare strings are\n"
        "what the uncoded residual is mostly made of."
    )


if __name__ == "__main__":
    main()
