"""Attributes of a source that the sector/role crossing cannot express.

:func:`covered.taxonomy.classify` reads a *role clause* -- ``FORMER CIA
DIRECTOR``, ``(D) ARIZONA`` -- and answers what kind of institution is
speaking. Two questions it structurally cannot answer come up as soon as
sources are counted rather than only typed:

**Is this a person or an institution?** ``The Washington Post said`` and
``Bob Woodward said`` are different epistemic acts, and a bare personal name
carries no role clause at all, so ``classify`` returns unknown for it. On the
sibling project's 100-article pilot only 12.8% of quoted sources got a sector
for exactly this reason -- the right tool answering a question it was not
asked. spaCy types the same strings at ~91%.

**If a person, how is the name gendered?** The gender composition of who gets
a microphone is a standard measure in this literature, and it is not derivable
from sector or role.

Both are deliberately kept off the critical path of the existing pipeline:
nothing here is imported by ``covered.pipeline``, and the axes are additive
with ``unknown`` defaults, so no previously computed number moves.

A warning that governs how the gender axis may be reported. Name-based gender
inference is a measurement of *how a name is gendered in aggregate data*, not
of anyone's identity -- which is what the underlying package's deliberately
awkward name, "nom quam gender" (name rather than gender), is there to keep in
view. It is valid for population-level description and invalid for labelling
any individual. The published benchmark (Santamaría & Mihaljević 2018) finds
error rates roughly ten times higher for Asian than European names, so any
result must be broken out by name origin or it is reporting the classifier.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from covered.config import SPACY_MODEL_FULL
from covered.speakers import is_nonperson

if TYPE_CHECKING:
    from spacy.language import Language

__all__ = ["GenderReading", "entity_form", "gender"]

# spaCy entity labels that stand for an institutional voice. GPE and NORP are
# included because metonymy is how broadcast attribution actually speaks --
# "Israel said", "the Kremlin announced", "Beijing denied" -- and treating
# those as anything but institutional sources would drop a real category.
# covered.attribution deliberately keeps a narrower {PERSON, ORG} set because
# it is matching syntactic subjects of cue verbs, which is a different job.
_PERSON_LABELS = frozenset({"PERSON"})
_ORG_LABELS = frozenset({"ORG", "GPE", "NORP", "FAC", "LOC"})


@dataclass(frozen=True)
class GenderReading:
    """How a name is gendered in reference data.

    Attributes:
        label: One of :data:`GENDERS`.
        p_female: Probability the name is used by women, or ``float("nan")``
            when the name is absent from the reference data. Kept alongside
            ``label`` because a thresholded label discards the distinction
            between a name that is genuinely ambiguous and one that is merely
            rare.
        sources: How many of the reference datasets contain the name. Zero
            means no evidence rather than balanced evidence.
        source: Which route produced the label -- ``name_model`` or ``none``.
    """

    label: str
    p_female: float
    sources: int
    source: str


@functools.lru_cache(maxsize=1)
def _model():
    """The name-gender reference model, loaded once."""
    import nomquamgender

    return nomquamgender.NBGC()


def entity_form(name: str, nlp: Language | None = None) -> tuple[str, str]:
    """Decide whether a source string names a person, an institution, neither.

    Args:
        name: Source string as written -- a speaker label, or the name a model
            reported as the claimer.
        nlp: Loaded spaCy pipeline. Loaded lazily from
            :data:`covered.config.SPACY_MODEL_FULL` when omitted; pass one in
            to avoid reloading per call.

    Returns:
        ``(form, source)`` where form is one of :data:`ENTITY_FORMS` and source
        records the rule that fired -- ``nonperson_list``, ``ner``, or ``none``.
        ``unnamed`` is a finding, not a failure: ``prosecutors``, ``officials``
        and ``sources`` are cited sources that cannot be resolved to a countable
        individual, which is precisely what makes them worth separating.
    """
    text = (name or "").strip()
    if not text:
        return "unknown", "none"
    if is_nonperson(text):
        return "unnamed", "nonperson_list"

    pipeline = nlp
    if pipeline is None:
        from covered.ner import load_nlp

        pipeline = load_nlp(SPACY_MODEL_FULL)
    labels = {ent.label_ for ent in pipeline(text).ents}
    if labels & _PERSON_LABELS:
        return "person", "ner"
    if labels & _ORG_LABELS:
        return "organization", "ner"
    return "unnamed", "ner"


def gender(name: str) -> GenderReading:
    """Read how a personal name is gendered in reference data.

    Only the given name is consulted, which is what the reference data indexes.
    Call this for sources already established to be people: an organisation's
    name has no given name and would either abstain or, worse, match a token
    that happens to look like one.

    Args:
        name: Personal name, e.g. ``"Volodymyr Zelenskyy"``.

    Returns:
        A :class:`GenderReading`. The label is ``unknown`` whenever the
        underlying model declines to classify -- it abstains on uncertainty
        rather than resolving coin flips, and on names it has never seen, such
        as the initials in ``J.D. Vance``.
    """
    text = (name or "").strip()
    if not text:
        return GenderReading("unknown", float("nan"), 0, "none")

    model = _model()
    annotated = model.annotate([text], as_df=True).iloc[0]
    p_female = float(annotated["p(gf)"])
    sources = int(annotated["sources"])
    # 'gm'/'gf'/'-' -- "gendered male/female", the package's framing, kept in
    # translation rather than flattened to "male"/"female" without comment.
    code = model.classify([text])[0]
    label = {"gf": "female", "gm": "male"}.get(code, "unknown")
    return GenderReading(
        label,
        p_female,
        sources,
        "name_model" if label != "unknown" else "none",
    )
