"""Generate every table, figure and inline number used in the manuscript.

Nothing in ``ms/covered.tex`` is typed by hand. Inline figures come from
``tabs/macros.tex`` as LaTeX commands, so re-running this after a lexicon
change, the dictionary tier, or validation updates the paper instead of
silently desynchronising it. The check is that the manuscript contains no bare
numerals outside the macro file.

    python scripts/make_paper_assets.py

Writes:
    tabs/macros.tex               every inline number, as \\newcommand
    tabs/sector_epistemic.tex     Table 1: sector x epistemic role
    tabs/coverage_diagnosis.tex   SI table: composition by string rarity
    figs/proximity_by_mode.pdf    Figure 1: direct access, bookings vs clips
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from covered import taxonomy
from covered.config import ANALYSIS_ROOT, INTERIM, TABLES

MS = ANALYSIS_ROOT / "tabs"
FIGS = ANALYSIS_ROOT / "figs"

DIRECT = ("principal", "participant", "eyewitness", "personal_experience")
MEASURED = frozenset({"lexicon", "form", "standing_rule", "name_title", "topic_needed"})
PROXIMITY = {
    **dict.fromkeys(DIRECT, "direct"),
    "spokesperson": "institutional",
    "expert": "expert",
    "commentator": "commentary",
    "popular_opinion": "commentary",
    "unresolved": "unresolved",
}

# Display names, in the order they should appear in the paper.
SECTOR_LABEL = {
    "government_executive": "Government, executive",
    "government_legislative": "Government, legislative",
    "government_subnational": "Government, subnational",
    "judicial_legal": "Judicial and legal",
    "law_enforcement": "Law enforcement",
    "military": "Military",
    "party_campaign": "Party and campaign",
    "business": "Business",
    "academic": "Academic",
    "think_tank": "Think tank",
    "nonprofit_advocacy": "Nonprofit and advocacy",
    "labor_union": "Labor union",
    "religious": "Religious",
    "media": "Media (other outlets)",
    "professional": "Professional",
    "entertainment_sport": "Entertainment and sport",
    "private_individual": "Private individual",
    "unknown": "Unknown",
}
ROLE_ORDER = [
    "principal",
    "spokesperson",
    "participant",
    "eyewitness",
    "personal_experience",
    "expert",
    "commentator",
    "popular_opinion",
    "unresolved",
]
ROLE_SHORT = {
    "principal": "Prin.",
    "spokesperson": "Spok.",
    "participant": "Part.",
    "eyewitness": "Eyew.",
    "personal_experience": "Exp'ce",
    "expert": "Expert",
    "commentator": "Comm.",
    "popular_opinion": "Pop.",
    "unresolved": "Unres.",
}


def _labelled_roles() -> pd.DataFrame:
    df = pd.read_csv(INTERIM / "role_strings.csv")
    g = df[df["staff_flag"] == "guest"].copy()
    labels = [taxonomy.classify(str(r)) for r in g["role_raw"]]
    g["sector"] = [x.sector for x in labels]
    g["epistemic_role"] = [x.epistemic_role for x in labels]
    g["measured"] = [x.epistemic_source in MEASURED for x in labels]
    return g


def _table_sector_epistemic(g: pd.DataFrame) -> dict[str, str]:
    """Table 1: sector by epistemic role, row-percentaged, unresolved kept."""
    ct = (
        g.pivot_table(
            index="sector",
            columns="epistemic_role",
            values="n_turns",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=ROLE_ORDER, fill_value=0)
        .reindex(index=[s for s in SECTOR_LABEL if s in g["sector"].unique()])
    )
    total = ct.sum(axis=1)
    pct = ct.div(total, axis=0) * 100

    # Normalised entropy of the epistemic distribution within each sector. This
    # is where the crossing does or does not earn its keep: a sector near 0 is
    # one whose epistemic role the sector already determines, so the second axis
    # carries nothing there. Reporting it per row stops an aggregate NMI from
    # implying the crossing works evenly across sectors.
    p = ct.div(total, axis=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    entropy = -terms.sum(axis=1) / np.log(len(ROLE_ORDER))

    lines = [
        r"\begin{table}[htbp]",
        # twelve columns overflow the text block at \small with default column
        # separation; the H column silently fell off the page edge.
        r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\caption{Institutional sector by epistemic role, CNN guest turns, "
        r"2000--2025. Row percentages; \% turns is each sector's share of all "
        r"guest turns. \emph{Unres.} is reported rather than dropped: it is the "
        r"share the role text does not settle. $H$ is the normalised entropy of "
        r"each row: sectors near zero are ones the sector alone already "
        r"determines, where the second axis adds nothing.}",
        r"\label{tab:crossing}",
        r"\begin{tabular}{l" + "r" * (len(ROLE_ORDER) + 2) + "}",
        r"\hline\hline",
        "Sector & "
        + " & ".join(ROLE_SHORT[r] for r in ROLE_ORDER)
        + r" & \% turns & $H$ \\",
        r"\hline",
    ]
    grand = float(g["n_turns"].sum())
    for i, sector in enumerate(pct.index):
        cells = " & ".join(f"{pct.loc[sector, r]:.1f}" for r in ROLE_ORDER)
        lines.append(
            f"{SECTOR_LABEL[sector]} & {cells} & "
            f"{100 * total[sector] / grand:.1f} & {entropy[i]:.2f} " + r"\\"
        )
    lines += [r"\hline\hline", r"\end{tabular}", r"\end{table}", ""]
    (MS / "sector_epistemic.tex").write_text("\n".join(lines), encoding="utf-8")

    # Normalised mutual information between the axes: 0 means independent (both
    # axes needed), 1 means the second is recoverable from the first.
    joint = ct.to_numpy() / ct.to_numpy().sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = float(np.where(joint > 0, joint * np.log(joint / (px @ py)), 0.0).sum())
    hx = float(-np.where(px > 0, px * np.log(px), 0.0).sum())
    hy = float(-np.where(py > 0, py * np.log(py), 0.0).sum())
    return {"NMI": f"{2 * mi / (hx + hy):.2f}"}


def _table_coverage(g: pd.DataFrame) -> None:
    """SI table: composition by how often the role string recurs."""
    diag = pd.read_csv(TABLES / "coverage_diagnosis.csv")
    sub = diag[diag["axis"] == "epistemic_role"]
    wide = sub.pivot_table(
        index="category", columns="quintile", values="share", aggfunc="sum"
    )
    for col in ("rarest", "commonest"):
        if col not in wide:
            wide[col] = 0.0
    wide["gap"] = (wide["rarest"] - wide["commonest"]) * 100
    wide = wide.sort_values("gap")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Epistemic composition of codable role strings, by how often "
        r"the string recurs. Quintiles of string frequency; percentages within "
        r"quintile. Categories over-represented among rare strings are the ones "
        r"the uncoded residual is mostly made of, and are therefore "
        r"under-counted in Table~\ref{tab:crossing}.}",
        r"\label{tab:rarity}",
        r"\begin{tabular}{lrrr}",
        r"\hline\hline",
        r"Epistemic role & Rarest \% & Commonest \% & Gap (pp) \\",
        r"\hline",
    ]
    for cat in wide.index:
        lines.append(
            f"{ROLE_SHORT.get(cat, cat)} & {wide.loc[cat, 'rarest'] * 100:.1f} & "
            f"{wide.loc[cat, 'commonest'] * 100:.1f} & "
            f"{wide.loc[cat, 'gap']:+.1f} " + r"\\"
        )
    lines += [r"\hline\hline", r"\end{tabular}", r"\end{table}", ""]
    (MS / "coverage_diagnosis.tex").write_text("\n".join(lines), encoding="utf-8")


def _figure_and_mode_numbers() -> dict[str, str]:
    """Figure 1 plus the booking/clip numbers quoted in the text."""
    ann = pd.read_csv(TABLES / "epistemic_annual.csv")
    ann = ann[(ann["staff_flag"] == "guest") & (ann["year"] < 2025)]
    ann = ann[ann["epistemic_source"].isin(sorted(MEASURED))]
    ann = ann[ann["proximity"] != "unresolved"]

    share = {}
    for mode in ("live", "clip"):
        m = ann[ann["mode"] == mode]
        wide = m.pivot_table(
            index="year", columns="proximity", values="n_turns", aggfunc="sum"
        ).fillna(0)
        share[mode] = wide["direct"] / wide.sum(axis=1)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for year in (2002, 2014):
        ax.axvline(year, color="0.75", linestyle="--", linewidth=1, zorder=0)
    ax.plot(
        share["clip"].index,
        share["clip"] * 100,
        marker="o",
        ms=3,
        color="#1b3a5c",
        label="Played clips",
    )
    ax.plot(
        share["live"].index,
        share["live"] * 100,
        marker="s",
        ms=3,
        color="#b5651d",
        label="Live bookings",
    )
    ax.set_ylabel("Direct-access share of turns (%)")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "proximity_by_mode.pdf")
    plt.close(fig)

    # The paper claims the gap holds in every year. Verify it rather than
    # assert it, and report the exception count if it does not.
    both = pd.DataFrame({"clip": share["clip"], "live": share["live"]}).dropna()
    exceptions = int((both["clip"] <= both["live"]).sum())
    gap = (both["clip"] - both["live"]) * 100
    print(
        f"  clip > live in {len(both) - exceptions}/{len(both)} years; "
        f"gap min {gap.min():.1f}pp, median {gap.median():.1f}pp"
    )
    if exceptions:
        print(f"  WARNING: {exceptions} year(s) reverse the claimed ordering")

    return {
        "ClipDirectLo": f"{share['clip'].min() * 100:.0f}",
        "ClipDirectHi": f"{share['clip'].max() * 100:.0f}",
        "LiveDirectLo": f"{share['live'].min() * 100:.0f}",
        "LiveDirectHi": f"{share['live'].max() * 100:.0f}",
        "ModeYears": f"{len(both)}",
        "ModeGapYears": f"{len(both) - exceptions}",
        "ModeGapMin": f"{gap.min():.0f}",
        "ModeGapMedian": f"{gap.median():.0f}",
    }


def main() -> None:
    MS.mkdir(parents=True, exist_ok=True)
    g = _labelled_roles()
    total = float(g["n_turns"].sum())

    def share(mask: pd.Series) -> float:
        return 100 * float(g.loc[mask, "n_turns"].sum()) / total

    macros = _table_sector_epistemic(g)
    _table_coverage(g)
    macros |= _figure_and_mode_numbers()

    # How strongly classifiability itself trends over the period. The paper
    # quotes this as the reason the over-time series cannot be interpreted.
    ann = pd.read_csv(TABLES / "epistemic_annual.csv")
    ann = ann[(ann["staff_flag"] == "guest") & (ann["year"] < 2025)]
    per_year = ann.groupby("year")["n_turns"].sum()
    measured = (
        ann[ann["epistemic_source"].isin(sorted(MEASURED))]
        .groupby("year")["n_turns"]
        .sum()
        .reindex(per_year.index)
        .fillna(0)
    )
    cov = (measured / per_year).to_numpy()
    macros["CodabilityCorr"] = (
        f"{np.corrcoef(per_year.index.to_numpy(), cov)[0, 1]:.2f}"
    )

    by_sector = g.groupby("sector")["n_turns"].sum()
    exec_share = 100 * by_sector.get("government_executive", 0) / total
    leg_share = 100 * by_sector.get("government_legislative", 0) / total
    macros |= {
        "GuestTurns": f"{int(total):,}",
        "DistinctRoles": f"{len(g):,}",
        "ExecShare": f"{exec_share:.1f}",
        "LegShare": f"{leg_share:.1f}",
        "ExecLegRatio": f"{exec_share / leg_share:.0f}",
        "PrivateShare": f"{share(g['sector'] == 'private_individual'):.1f}",
        "MediaShare": f"{share(g['sector'] == 'media'):.1f}",
        "UnknownShare": f"{share(g['sector'] == 'unknown'):.1f}",
        "MeasuredShare": f"{share(g['measured']):.1f}",
        "CommentatorShare": f"{share(g['epistemic_role'] == 'commentator'):.1f}",
        "PrincipalShare": f"{share(g['epistemic_role'] == 'principal'):.1f}",
    }

    lines = [
        "% Generated by scripts/make_paper_assets.py -- do not edit.",
        "% Every inline number in the manuscript comes from here, so the paper",
        "% cannot drift out of step with the data.",
    ]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (MS / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {MS}/macros.tex with {len(macros)} values")
    for k, v in sorted(macros.items()):
        print(f"  {k:<20} {v}")
    print(f"wrote {MS}/sector_epistemic.tex, {MS}/coverage_diagnosis.tex")
    print(f"wrote {FIGS}/proximity_by_mode.pdf")


if __name__ == "__main__":
    main()
