"""LLM validation tier: stratified sampling + an LLM first-pass annotator.

The rule-based / spaCy extraction runs over the full corpus; this module draws a
stratified sample, has an LLM produce gold labels for the three extraction tasks,
and (via :mod:`covered.metrics`) turns measured precision/recall into an
error-adjusted HHI. LLM labels are a first pass, not ground truth — a
human-adjudicated subset estimates LLM-vs-human agreement.

The Anthropic call is gated on ``ANTHROPIC_API_KEY`` and the optional ``llm``
extra; the sampling logic has no such dependency and is unit-tested directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

__all__ = ["LLM_MODEL", "TASK_RUBRICS", "annotate_segment", "stratified_sample"]

# Pinned for reproducibility; record alongside outputs in data/validation/.
LLM_MODEL = "claude-opus-4-8"

TASK_RUBRICS: dict[str, str] = {
    "speakers": (
        "List every on-air speaker turn in this CNN transcript segment. For each, "
        "give the speaker's name as labeled, their role/title if present, and whether "
        "they are CNN staff (host/anchor/correspondent/analyst) or an external guest. "
        "Exclude stage directions and UNIDENTIFIED speakers from the named list."
    ),
    "attributions": (
        "List every third-party individual quoted or paraphrased in the body text "
        "(e.g. 'Senator Smith said ...'). Give the person's name and the cue verb. "
        "Include direct quotes and paraphrases; exclude the on-air speaker quoting "
        "themselves; classify organizations separately from individuals."
    ),
    "entities": (
        "For the given surface name and its sentence context, return the canonical "
        "individual it refers to, or 'ambiguous' if it cannot be determined."
    ),
}


def stratified_sample(
    df: pd.DataFrame,
    strata: Sequence[str],
    per_stratum: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Draw up to ``per_stratum`` rows from each stratum (deterministic by ``seed``).

    Strata smaller than ``per_stratum`` contribute all their rows.
    """
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby(list(strata)):
        n = min(per_stratum, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def annotate_segment(
    text: str,
    task: str,
    client: object | None = None,
    model: str = LLM_MODEL,
) -> str:
    """Have the LLM produce gold labels for one segment under a task rubric.

    Returns the model's raw text. Requires the ``anthropic`` package (the ``llm``
    extra) and ``ANTHROPIC_API_KEY``. The caller adjudicates and scores against
    the rule-based extraction in :mod:`covered.metrics`.
    """
    if task not in TASK_RUBRICS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASK_RUBRICS)}")
    import anthropic  # lazy: optional dependency, needs an API key at call time

    anthropic_client = client or anthropic.Anthropic()
    response = anthropic_client.messages.create(  # type: ignore[union-attr]
        model=model,
        max_tokens=4000,
        system=TASK_RUBRICS[task],
        messages=[{"role": "user", "content": text}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")
