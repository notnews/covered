"""spaCy pipeline loading, cached per model name.

The model version is pinned in :mod:`covered.config` because a model upgrade
changes NER output and therefore the HHI series.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.language import Language

__all__ = ["load_nlp"]


@functools.lru_cache(maxsize=2)
def load_nlp(model_name: str) -> Language:
    """Load (and cache) a spaCy model by name.

    The full pipeline (tagger, parser, lemmatizer, ner) is required: attribution
    matching relies on dependency parses and lemmas, not just NER.
    """
    import spacy

    return spacy.load(model_name)
