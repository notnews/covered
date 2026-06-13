"""Cited-source (quotation) HHI trend, per-year reservoir sample.

Attribution extraction is spaCy-bound, so a full-corpus pass is hours; instead
we draw a per-year reservoir sample (memory-safe), run the dependency-matcher
attribution extractor, and report the annual cited-individual concentration.

    python scripts/trend_attributions.py   # writes outputs/tables/hhi_attrib_annual_sampled.csv
"""

import random
from collections import defaultdict

import pandas as pd

from covered import acquire, ner, pipeline, schema
from covered.config import PROCESSED, TABLES

QUOTA = 500  # segments sampled per year
SEED = 20250611


def main() -> None:
    rng = random.Random(SEED)
    res: dict[int, list[dict]] = defaultdict(list)
    seen: dict[int, int] = defaultdict(int)
    cols = ["url", "text", "year", "uid", "month", "date"]

    for p in acquire.raw_files():
        n = 0
        for chunk in pd.read_csv(p, dtype=str, keep_default_na=False, chunksize=20000):
            chunk = schema.validate_csv(chunk).dropna(subset=["year"])
            for r in chunk[cols].to_dict("records"):
                y = int(r["year"])
                seen[y] += 1
                if len(res[y]) < QUOTA:
                    res[y].append(r)
                else:
                    j = rng.randint(0, seen[y] - 1)
                    if j < QUOTA:
                        res[y][j] = r
            n += len(chunk)
        print(f"[{p.name}] scanned {n:,}", flush=True)

    sample = pd.DataFrame([r for rows in res.values() for r in rows])
    print(f"sampled {len(sample):,} segments across {sample['year'].nunique()} years", flush=True)

    nlp = ner.load_nlp("en_core_web_sm")
    atts = pipeline.build_attributions(sample, nlp)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    atts.to_parquet(PROCESSED / "attrib_sample.parquet", index=False)

    person = pipeline.annual_hhi_attributions(atts, "PERSON")
    org = pipeline.annual_hhi_attributions(atts, "ORG")
    ann = pd.concat([person, org], ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    dest = TABLES / "hhi_attrib_annual_sampled.csv"
    ann.to_csv(dest, index=False)
    print(f"extracted {len(atts):,} attributions -> {dest}", flush=True)


if __name__ == "__main__":
    main()
