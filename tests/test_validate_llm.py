"""Tests for stratified sampling (LLM call itself is gated on an API key)."""

import pandas as pd

from covered import validate_llm


def _frame() -> pd.DataFrame:
    rows = []
    for era in ("era1", "era2", "era3"):
        for i in range(10):
            rows.append({"uid": f"{era}.{i}", "era_id": era, "show_code": "acd"})
    return pd.DataFrame(rows)


def test_stratified_sample_respects_per_stratum() -> None:
    sample = validate_llm.stratified_sample(_frame(), strata=["era_id"], per_stratum=4, seed=1)
    counts = sample["era_id"].value_counts().to_dict()
    assert counts == {"era1": 4, "era2": 4, "era3": 4}


def test_stratified_sample_is_deterministic() -> None:
    a = validate_llm.stratified_sample(_frame(), strata=["era_id"], per_stratum=3, seed=42)
    b = validate_llm.stratified_sample(_frame(), strata=["era_id"], per_stratum=3, seed=42)
    assert list(a["uid"]) == list(b["uid"])


def test_stratified_sample_caps_at_stratum_size() -> None:
    small = pd.DataFrame([{"uid": "x", "era_id": "era1"}])
    sample = validate_llm.stratified_sample(small, strata=["era_id"], per_stratum=5, seed=0)
    assert len(sample) == 1
