"""Label the long tail of role strings once, offline, into a frozen dictionary.

The keyword lexicons generalise, which is what makes them auditable, but a long
tail of role strings has nothing to generalise from -- ``PRINCESS DIANA'S LOVER
1986-91``, ``DIRECTOR EMERITUS, COLUMBUS ZOO``, ``HALEIGH'S BABY-SITTER``. This
script asks an LLM to place those on the same two axes, working from the same
codebook the lexicons encode, and writes the answers to
``data/reference/role_dictionary.csv``.

That CSV is the deliverable. It is committed, and ``taxonomy.classify`` reads it
by exact match, so:

* the production pipeline stays deterministic and needs no API key;
* the dictionary is consulted only where every keyword rule declined, so the
  auditable layer always wins;
* labels from it are tagged ``dictionary``, so tables can report what the tier
  contributed and validation can score it separately from the rules.

    export ANTHROPIC_API_KEY=...
    python scripts/label_roles_llm.py --limit 6000          # write the dictionary
    python scripts/label_roles_llm.py --limit 6000 --dry-run  # cost estimate only

Ranked by turn count, so the budget buys the most corpus coverage. Labelling the
top ~5,900 currently-unknown strings takes the unknown share from 0.255 to about
0.15; the top 1,000 takes it to about 0.18.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import pandas as pd

from covered import taxonomy
from covered.config import INTERIM, REFERENCE

MODEL = "claude-opus-4-8"  # pinned; record alongside the output
BATCH = 60  # role strings per request

CODEBOOK = f"""You are coding the role clause that US cable-news transcripts attach to a
speaker label, as in "JANE DOE, UKRAINIAN REFUGEE:". Place each role on two
independent axes. Judge only from the role text; do not speculate about who the
person is beyond what it says.

SECTOR -- the institutional home. One of:
{", ".join(taxonomy.SECTORS)}

  government_executive   presidents, ministers, cabinet, agency heads, press secretaries
  government_legislative senators, representatives, MPs
  government_subnational governors, mayors, city and county officials
  judicial_legal         judges, prosecutors, defence attorneys, counsel
  law_enforcement        police, FBI, sheriffs, detectives
  military               serving and retired armed forces
  party_campaign         strategists, consultants, campaign staff, candidates
  business               executives, founders, owners, market analysts
  academic               professors, universities, researchers, historians
  think_tank             policy institutes and their fellows
  nonprofit_advocacy     NGOs, charities, activists
  labor_union            union officers and representatives
  religious              clergy and theologians
  media                  journalists and outlets OTHER than the network itself
  professional           credentialed non-academic: doctors, psychologists,
                         meteorologists, engineers, pilots
  entertainment_sport    performers, artists, chefs, athletes, coaches
  private_individual     people with no institutional standing: relatives,
                         neighbours, witnesses, victims, defendants, students,
                         refugees, volunteers, supporters
  unknown                the role text does not say

EPISTEMIC ROLE -- why they are being cited. One of:
{", ".join(taxonomy.EPISTEMIC_ROLES)}

  principal            the actor whose own conduct or decisions are the news
  spokesperson         speaks on behalf of an institution, not for themselves
  expert               credentialed knowledge bearing on the topic
  commentator          opinion, strategy or prediction; no privileged access
  participant          a direct role in the events being reported
  eyewitness           observed the events without being party to them
  personal_experience  personally affected; speaks from that standing
  popular_opinion      cited as a member of the public
  unresolved           the role text does not say

Rules that matter:
* "FORMER X" loses the standing X conferred. A former officeholder is a
  commentator, not a principal or spokesperson.
* Someone tied to a NAMED party in the story ("ATTORNEY FOR X", "FRIEND OF X")
  is a participant or personal_experience, not a commentator.
* Prefer "unknown"/"unresolved" over a guess. A wrong label is worse than a
  missing one, because a missing one is countable.

Return ONLY a JSON array, one object per input, in the same order:
[{{"role": "<verbatim input>", "sector": "...", "epistemic_role": "..."}}]"""


def _pending(limit: int) -> pd.DataFrame:
    """Currently-unknown guest role strings, ranked by turns."""
    df = pd.read_csv(INTERIM / "role_strings.csv")
    g = df[df["staff_flag"] == "guest"].copy()
    g["sector"] = [taxonomy.classify(str(r)).sector for r in g["role_raw"]]
    un = g[g["sector"] == "unknown"].sort_values("n_turns", ascending=False)
    already = set(taxonomy._role_dictionary())
    un = un[~un["role_raw"].astype(str).str.lower().isin(already)]
    return un.head(limit)


def _label_batch(client: object, roles: list[str]) -> list[dict[str, str]]:
    numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(roles))
    resp = client.messages.create(  # type: ignore[attr-defined]
        model=MODEL,
        max_tokens=8000,
        system=CODEBOOK,
        messages=[{"role": "user", "content": numbered}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=6000, help="role strings to label")
    ap.add_argument("--dry-run", action="store_true", help="report scope, call nothing")
    args = ap.parse_args()

    pending = _pending(args.limit)
    if pending.empty:
        print("nothing pending")
        return

    df = pd.read_csv(INTERIM / "role_strings.csv")
    tot = df[df["staff_flag"] == "guest"]["n_turns"].sum()
    covered_turns = int(pending["n_turns"].sum())
    print(f"{len(pending):,} role strings pending, {covered_turns:,} turns")
    print(f"  = {covered_turns / tot:.3f} of guest turns")
    print(f"  {-(-len(pending) // BATCH):,} requests of {BATCH}")
    if args.dry_run:
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")
    import anthropic  # lazy: optional 'llm' extra

    client = anthropic.Anthropic()

    dest = REFERENCE / "role_dictionary.csv"
    new = not dest.exists()
    with dest.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            fh.write(
                "# Per-string role labels for the tail no keyword rule reaches.\n"
                f"# Written by scripts/label_roles_llm.py using {MODEL} against the\n"
                "# same codebook the lexicons encode. Frozen and committed: the model\n"
                "# never runs in the production path, and taxonomy.classify consults\n"
                "# this only where every keyword rule declined, tagging the result\n"
                "# 'dictionary' so its contribution stays separable in every table.\n"
            )
            w.writerow(["role_raw", "sector", "epistemic_role", "n_turns", "model"])
        roles = pending["role_raw"].astype(str).tolist()
        turns = dict(zip(roles, pending["n_turns"], strict=True))
        done = 0
        for i in range(0, len(roles), BATCH):
            chunk = roles[i : i + BATCH]
            for rec in _label_batch(client, chunk):
                role = str(rec.get("role", "")).strip()
                sector = str(rec.get("sector", "")).strip().lower()
                erole = str(rec.get("epistemic_role", "")).strip().lower()
                if role in turns and sector in taxonomy.SECTORS:
                    w.writerow([role, sector, erole, turns[role], MODEL])
                    done += 1
            fh.flush()
            print(f"  {min(i + BATCH, len(roles)):,}/{len(roles):,} sent", flush=True)
    print(f"wrote {done:,} labels -> {dest}")


if __name__ == "__main__":
    main()
