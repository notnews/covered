# covered — who gets covered in cable news

`covered` measures **whose voices occupy the news** — at scale, across a
quarter-century of transcripts. It asks a simple question with antitrust tools:
how *concentrated* is news coverage among a small set of elite voices, and has
that changed over time?

Two voices are tracked **separately**, because they answer different questions:

- **(a) Speakers** — who is given a microphone, parsed from `NAME, ROLE:`
  speaker labels. Further split into **live** appearances vs. **played clips**
  (`(BEGIN VIDEO CLIP)…`): *whom the network books* vs. *whose words it plays*.
- **(b) Cited sources** — third-party individuals quoted or paraphrased in body
  text ("Senator Smith said…"), extracted with NLP.

The ruler is the **Herfindahl–Hirschman Index (HHI)** — the standard measure of
market concentration (sum of squared shares; `1/HHI` = the effective number of
equally-loud voices). It's reported per year alongside top-k share, normalized
HHI, entropy, and distinct-source counts.

First corpus: **CNN, 2000–2025, 336,902 segments** (Harvard Dataverse,
`doi:10.7910/DVN/ISDPJU`). The design generalizes to the sister
[notnews](https://github.com/notnews) corpora (Fox, MSNBC, …).

## Headline findings (CNN, 2000–2025)

- **Live booking never concentrated.** External-guest HHI is a flat ~0.001–0.003
  for 25 straight years (≈500 effective voices). CNN keeps booking a broad set.
- **Clip airtime did — and it's the president.** Clip-only HHI jumped ~10×
  during 2017–2020 (0.004 → **0.044**; top-10 clip sources took ~31% of clip
  airtime). The top clip source is the sitting president *every year*; in
  2017–2020 Trump alone took 9.6–10.7% of all guest turns.
- So the "narrowing marketplace" is **not** narrower booking — it's the network
  devoting a historically concentrated share of *played-clip* airtime to one
  principal's words. Clips also grew from ~28% of guest turns (2000s) to ~50%
  (2018–2025).
- **Quotation is a narrower elite than the microphone** (first slice): cited
  individuals run more concentrated than booked guests.

See `outputs/tables/` for the per-year series behind these.

## Data

The corpus is **not** in this repo — it is access-restricted (research use
only) on Harvard Dataverse, `doi:10.7910/DVN/ISDPJU`, as eight compressed files
(`cnn-1…6.7z`, `cnn-7/8.csv.gz`), 2.5 GB.

```bash
# 1. Request access on Dataverse and get an API token, then:
export DATAVERSE_API_TOKEN=...        # never commit it

# 2. Download + extract all eras into data/raw/ (writes a checksum manifest)
make acquire
```

`acquire` writes `data/reference/manifest.json` pinning the dataset version and
SHA-256 of every file; `.7z` archives are extracted to CSV automatically.

## Quickstart

```bash
make setup            # uv venv (Python 3.12) + deps + spaCy models
make check            # ruff + mypy + pytest (run on fixtures, no data needed)
make acquire          # download the corpus (needs DATAVERSE_API_TOKEN)

# full 25-year trends -> outputs/tables/
python scripts/trend_speakers.py        # speaker HHI: all / live / clip (regex, minutes)
python scripts/trend_attributions.py    # cited-source HHI, per-year sample (spaCy, ~40 min)
```

## How it works (pipeline)

```
download → parse speakers → extract citations → resolve entities → HHI → figures
 acquire     speakers.py      attribution.py       entities.py      hhi.py  figures.py
```

Every extracted record (a turn or a citation) carries full **provenance** —
parsed show slug, inferred host, dateline, headline/subhead, source URL, and
in-text character offsets — so any count can be traced back to its source
segment and audited. Concentration is reported with an **error-adjusted** HHI: a
stratified LLM-validation tier (`validate_llm.py`, `metrics.py`) estimates
extraction precision/recall and propagates that uncertainty into the series.

## Layout

```
covered/
  src/covered/        # the package (parsing, extraction, resolution, HHI, CLI)
  scripts/            # full-corpus trend drivers (memory-safe, chunked)
  tests/              # unit + golden-file + end-to-end (small spaCy model)
  data/
    raw/              # downloaded corpus            (gitignored)
    interim/ processed/  # parsed tables, registries (gitignored)
    reference/        # alias tables, role/cue lexicons, show map, manifest (tracked)
    validation/       # LLM/human gold labels        (tracked)
  outputs/tables/     # the annual HHI series        (tracked)
  outputs/figures/    # plots                        (gitignored)
```

## Reproducibility

- `uv`-pinned Python 3.12; all deps and the spaCy model version pinned (NER
  changes change the HHI).
- `ruff` (lint + format), `mypy`, `pytest` via `make check`; `pre-commit` hooks;
  GitHub Actions CI runs the same on fixtures.

## Caveats (honest)

- Entity resolution curates the high-frequency head (presidents, anchors) and
  leaves ambiguous bare surnames explicitly unresolved rather than over-merging.
- The role lexicon still mislabels some CNN contributors (meteorologists,
  business/weather) as guests — affects the *live* top-name labels, not the
  (diffuse) live HHI.
- A few NER/parse artifacts in the long tail (bare first names; a stray `OK:`)
  are what the LLM-validation tier exists to quantify and correct.
- 2025 is partial (Jan–Mar) — excluded from headline trends.
