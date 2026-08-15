"""Gender for the people the corpus names most, from a frozen Wikidata cache.

Name-based inference reads the name; this reads the person. For the head of
the distribution -- heads of state, cabinet officials, the network's own
recurring roster -- Wikidata records gender as a curated fact, so it is both
more accurate than any name model and, more importantly, *independent* of one.
Two routes to the same quantity sharing no code is what makes disagreement
between them informative rather than merely irritating: agreement is evidence,
and every disagreement is a case worth reading.

**The cache is the interface.** :func:`resolve` never touches the network. It
reads ``data/reference/wikidata_persons.csv``, which is committed, so a run is
reproducible offline and byte-identical next year regardless of what Wikidata
does in the meantime. :func:`refresh` is the only function that makes requests
and is meant to be run deliberately, its output reviewed, and the result
committed -- the same frozen-tier pattern the role dictionary uses.

**Disambiguation is the risk, not the lookup.** A search for a name returns
whatever shares it: querying "Volodymyr Zelenskyy" offers the president, a
podcast episode about him, and his inauguration ceremony. Anything that is not
an instance of human (``P31 -> Q5``) is discarded, and the QID is recorded in
the cache so that a wrong match is visible and correctable by hand rather than
buried in a probability.

**Gender is not forced into two values.** ``P21`` most often reads male or
female, but it does not only read those. Values outside the pair are recorded
verbatim in ``gender_qid`` and labelled ``unknown`` rather than being rounded
into the nearer bucket, because silently binarising a person's recorded gender
is a worse error than declining to code it.
"""

from __future__ import annotations

import csv
import functools
from dataclasses import dataclass

from covered.config import REFERENCE
from covered.entities import normalize_name

__all__ = ["WikidataPerson", "cache_path", "fetch", "refresh", "resolve"]

_API = "https://www.wikidata.org/w/api.php"
# Wikidata asks that automated clients identify themselves and a contactable
# project; an unidentified bulk client can be rate-limited or blocked.
_UA = "covered-research/0.1 (https://github.com/notnews/covered; academic use)"

_HUMAN = "Q5"
_SEX_OR_GENDER = "P21"
_INSTANCE_OF = "P31"
_OCCUPATION = "P106"

# Only the two values that map cleanly onto the GENDERS vocabulary. Everything
# else is preserved as a QID rather than coerced -- see the module docstring.
_GENDER_QIDS = {"Q6581097": "male", "Q6581072": "female"}

_BATCH = 50  # wbgetentities' documented maximum ids per call
_RETRIES = 5
_BACKOFF = 1.0  # seconds, doubled per retry

_FIELDS = ("name_norm", "qid", "label", "gender", "gender_qid", "occupation")


@dataclass(frozen=True)
class WikidataPerson:
    """One resolved person.

    Attributes:
        name_norm: Normalised name used as the cache key.
        qid: Wikidata item id, recorded so a wrong match can be spotted.
        label: Wikidata's English label, for eyeballing the match.
        gender: One of :data:`covered.taxonomy.GENDERS`.
        gender_qid: Raw ``P21`` value, non-empty even when gender is unknown.
        occupation: First ``P106`` QID, or empty.
    """

    name_norm: str
    qid: str
    label: str
    gender: str
    gender_qid: str
    occupation: str


def cache_path():
    """Path to the committed lookup table."""
    return REFERENCE / "wikidata_persons.csv"


@functools.lru_cache(maxsize=1)
def _cache() -> dict[str, WikidataPerson]:
    path = cache_path()
    if not path.exists():
        return {}
    out: dict[str, WikidataPerson] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            key = (row.get("name_norm") or "").strip()
            if key:
                out[key] = WikidataPerson(
                    key,
                    (row.get("qid") or "").strip(),
                    (row.get("label") or "").strip(),
                    (row.get("gender") or "unknown").strip(),
                    (row.get("gender_qid") or "").strip(),
                    (row.get("occupation") or "").strip(),
                )
    return out


def resolve(name: str) -> WikidataPerson | None:
    """Look a name up in the frozen cache. Never makes a request.

    Args:
        name: Personal name as written, e.g. ``"Volodymyr Zelenskyy"``.

    Returns:
        The cached person, or ``None`` when the name was never resolved. A miss
        is a normal outcome: only the head of the distribution is worth curating,
        and the tail is left to the name model.
    """
    return _cache().get(normalize_name(name or ""))


def fetch(names: list[str], *, session=None) -> list[WikidataPerson]:
    """Query Wikidata for a batch of names. Makes network requests.

    Resolution is two-pass rather than per-name. A naive implementation asks
    "is this a human?" once per search candidate, which is up to seven requests
    per name and earns a 429 within a couple of hundred names. Here every name
    is searched once, then all candidates are fetched fifty at a time -- the
    limit ``wbgetentities`` accepts -- which is roughly a tenth of the traffic.

    Args:
        names: Personal names to resolve.
        session: Optional ``requests.Session`` for connection reuse and tests.

    Returns:
        One entry per name that resolved to a human. Names that matched nothing,
        or matched only non-human items, are omitted rather than returned with
        empty fields, so the caller can see the coverage shortfall.
    """
    import requests

    http = session or requests.Session()
    http.headers.update({"User-Agent": _UA})

    candidates: dict[str, list[str]] = {}
    for name in names:
        candidates[name] = _search(http, name)

    wanted = sorted({qid for qids in candidates.values() for qid in qids})
    entities: dict[str, dict] = {}
    for start in range(0, len(wanted), _BATCH):
        entities |= _entities(http, wanted[start : start + _BATCH])

    found: list[WikidataPerson] = []
    for name, qids in candidates.items():
        key = normalize_name(name)
        for qid in qids:
            entity = entities.get(qid, {})
            if _HUMAN not in _qid_values(entity, _INSTANCE_OF):
                continue
            label = entity.get("labels", {}).get("en", {}).get("value", "")
            if not _label_matches(label, key):
                continue
            found.append(_person(entity, qid, key))
            break
    return found


def _fold(text: str) -> str:
    """Normalise a name and strip diacritics, for comparison only."""
    from unidecode import unidecode

    return unidecode(normalize_name(text))


def _label_matches(label: str, key: str) -> bool:
    """Whether a candidate's label really names the person who was cited.

    Taking the first human search hit is not good enough, and the failure is
    quiet: it returned Jack Kelly for John Kelly, Nina Appel for Nina Schick,
    and a neuroscientist for a CNN meteorologist -- roughly 3% of a 138-person
    head, each one a confidently wrong gender rather than a missing one.

    Middle names and initials are tolerated because Wikidata prefers the full
    legal form ("James R. Clapper", "Brendan F. Boyle") where a transcript uses
    the everyday one. An unlabelled item is rejected: it cannot be checked, and
    the name model covers whatever is turned away here.

    Diacritics are folded, and skipping that step is worse than not checking at
    all. CNN writes "Aleksandar Vucic"; Wikidata labels him "Aleksandar Vučić".
    Comparing those literally rejects the actual president of Serbia and then
    accepts the next candidate down, an unrelated namesake carrying a middle
    initial -- a tightened guard that quietly swaps a right answer for a wrong
    one.

    Args:
        label: Wikidata's English label for the candidate.
        key: Normalised name as cited in the transcript.

    Returns:
        True when the label names the same person.
    """
    if not label:
        return False
    got = _fold(label).split()
    want = _fold(key).split()
    if not got or not want:
        return False
    if got == want:
        return True
    # First and last must agree exactly; anything between them may be dropped.
    return got[0] == want[0] and got[-1] == want[-1] and len(want) <= len(got)


def _get(http, params: dict) -> dict:
    """One API call, retrying politely when Wikidata asks us to slow down."""
    import time

    for attempt in range(_RETRIES):
        resp = http.get(_API, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(_BACKOFF * (2**attempt))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Wikidata kept returning 429; slow down or try later")


def _search(http, name: str) -> list[str]:
    """Candidate item ids for ``name``, best match first."""
    payload = _get(
        http,
        {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json",
            "limit": 5,
            "type": "item",
        },
    )
    return [hit["id"] for hit in payload.get("search", []) if hit.get("id")]


def _entities(http, qids: list[str]) -> dict[str, dict]:
    if not qids:
        return {}
    payload = _get(
        http,
        {
            "action": "wbgetentities",
            "ids": "|".join(qids),
            "props": "claims|labels",
            "languages": "en",
            "format": "json",
        },
    )
    return payload.get("entities", {})


def _qid_values(entity: dict, prop: str) -> list[str]:
    out = []
    for claim in entity.get("claims", {}).get(prop, []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        if isinstance(value, dict) and value.get("id"):
            out.append(value["id"])
    return out


def _person(entity: dict, qid: str, key: str) -> WikidataPerson:
    gender_qids = _qid_values(entity, _SEX_OR_GENDER)
    gender_qid = gender_qids[0] if gender_qids else ""
    occupations = _qid_values(entity, _OCCUPATION)
    return WikidataPerson(
        key,
        qid,
        entity.get("labels", {}).get("en", {}).get("value", ""),
        _GENDER_QIDS.get(gender_qid, "unknown"),
        gender_qid,
        occupations[0] if occupations else "",
    )


def refresh(names: list[str], *, session=None) -> int:
    """Resolve ``names`` and rewrite the committed cache. Makes requests.

    Existing rows are kept, so a hand-corrected entry survives a refresh and
    only genuinely new names are queried. Review the diff before committing:
    the point of freezing this file is that a human has looked at it.

    Args:
        names: Personal names to add.
        session: Optional ``requests.Session``.

    Returns:
        How many new people were added.
    """
    existing = dict(_cache())
    before = len(existing)
    wanted = [n for n in names if normalize_name(n or "") not in existing]
    for person in fetch(wanted, session=session):
        existing[person.name_norm] = person
    added = len(existing) - before

    path = cache_path()
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(
            "# Gender for frequently cited people, resolved from Wikidata P21 and\n"
            "# frozen so runs are reproducible offline. Regenerate with\n"
            "# covered.wikidata.refresh(), then read the diff before committing:\n"
            "# check qid/label actually name the person meant. A gender_qid that\n"
            "# is neither Q6581097 nor Q6581072 is recorded, not coerced.\n"
        )
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        for key in sorted(existing):
            person = existing[key]
            writer.writerow({f: getattr(person, f) for f in _FIELDS})

    _cache.cache_clear()
    return added
