"""Central configuration: paths, corpus epochs, seeds, pinned versions.

Keeping these in one place makes every downstream output traceable to a fixed
set of constants and lets tests reference the same boundaries the pipeline uses.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# --- paths -----------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent


def _analysis_root() -> Path:
    """Locate the analysis checkout that holds ``data/`` and ``outputs/``.

    Walking up from ``__file__`` only works for an editable install inside
    the checkout. From a built wheel the package lives in ``site-packages``,
    where no ``data/`` exists, so fall back to ``COVERED_ANALYSIS_ROOT`` and
    then to the working directory (the CLI is run from the analysis root).

    Returns:
        Root directory containing ``data/`` and ``outputs/``.
    """
    env = os.environ.get("COVERED_ANALYSIS_ROOT")
    if env:
        return Path(env).resolve()
    checkout = PACKAGE_ROOT.parent.parent
    if (checkout / "data" / "reference").is_dir():
        return checkout
    return Path.cwd()


ANALYSIS_ROOT = _analysis_root()
DATA = ANALYSIS_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REFERENCE = DATA / "reference"
VALIDATION = DATA / "validation"
OUTPUTS = ANALYSIS_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

# --- reproducibility -------------------------------------------------------
SEED = 20250611

# spaCy model used for NER/attribution. Pinned because a model upgrade changes
# entity output and therefore the HHI series.
SPACY_MODEL_FULL = "en_core_web_lg"  # full-corpus throughput
SPACY_MODEL_ACCURATE = "en_core_web_trf"  # validation-slice accuracy

# Dataverse dataset, pinned to a specific published version for reproducibility.
DATAVERSE_DOI = "doi:10.7910/DVN/ISDPJU"
DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"
DATAVERSE_VERSION = "LATEST"  # override with a numbered version once frozen

# --- corpus epochs ---------------------------------------------------------
# The CNN HTML (and therefore the stored ``text`` field) changed format twice.
# These boundaries align exactly with the published CSV file splits.
CORPUS_START = date(2000, 1, 1)
CORPUS_END = date(2025, 3, 15)

# (era_id, inclusive_start) ordered by start date.
ERA_STARTS: tuple[tuple[str, date], ...] = (
    ("era1", date(2000, 1, 1)),  # old h2/h3/h4 markup (cnn-1..cnn-4)
    ("era2", date(2002, 9, 17)),  # cnnBodyText CSS classes (cnn-5..cnn-6)
    (
        "era3",
        date(2014, 6, 18),
    ),  # modern /show/<code>/date/.../segment slug (cnn-7..cnn-8)
)
ERA_IDS: tuple[str, ...] = tuple(eid for eid, _ in ERA_STARTS)
