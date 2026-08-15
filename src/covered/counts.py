"""How many sources a story credits, and why that is a range not a number.

The first question asked of a story's sourcing is the simplest one: how many
people did the reporting actually rest on. It is also the one the data cannot
answer exactly, because journalism has a standard construction for citing a
source while withholding how many there were. "Officials told CNN" is one
citation covering an unknown number of people; "sources say" could be two
people or ten, and nothing in the transcript distinguishes those worlds.

So the count is *partially identified*. Rather than pick a convention -- count
each unnamed citation as one person, or ignore them -- this reports the bounds
the evidence actually supports:

    lower = max(distinct named sources, 1 if any unnamed citation)
    upper = distinct named sources + number of unnamed citations

The lower bound allows every unnamed citation to refer to someone already
counted by name, which happens constantly: a story quotes Jake Sullivan and
then writes "officials said". The upper bound allows each to be a new person.
When a story has only unnamed citations the lower bound is 1, because a cited
source is at least one person even when unidentifiable.

This is the same move :mod:`covered` already makes for the uncoded residual of
the taxonomy, with one difference worth stating: there, the truth exists and is
merely unobserved, so better coding narrows the bound. Here the transcript
genuinely does not contain the number, and no amount of better method will
recover it. The honest response is to report the width, and if it swamps the
between-story variation, to say the measure cannot support the comparison.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

__all__ = ["SourceCount", "bound_is_informative", "count_sources"]

# Entity forms that name a countable source. "unnamed" is deliberately absent:
# that is the whole point of the bound.
_COUNTABLE = frozenset({"person", "organization"})


@dataclass(frozen=True)
class SourceCount:
    """Sources credited by one story.

    Attributes:
        n_person: Distinct named people credited.
        n_organization: Distinct named institutions credited.
        n_unnamed_citations: Citations to a source that cannot be resolved to
            an individual. A count of *citations*, not of sources -- counting
            them as sources is precisely the error the bounds exist to avoid.
        lower: Fewest people the story can rest on.
        upper: Most people the story can rest on.
    """

    n_person: int
    n_organization: int
    n_unnamed_citations: int
    lower: int
    upper: int

    @property
    def named(self) -> int:
        """Distinct named sources, people and institutions together."""
        return self.n_person + self.n_organization

    @property
    def width(self) -> int:
        """How much of the count the transcript does not determine."""
        return self.upper - self.lower


def count_sources(citations: Iterable[tuple[str, str]]) -> SourceCount:
    """Count the sources one story credits.

    Args:
        citations: One ``(key, entity_form)`` pair per *citation*, not per
            source -- repeated citations of the same person are expected and
            are deduplicated here. ``key`` should already be normalised (see
            :func:`covered.entities.normalize_name`); ``entity_form`` is one of
            :data:`covered.taxonomy.ENTITY_FORMS`.

    Returns:
        The counts and the bounds they support.
    """
    people: set[str] = set()
    orgs: set[str] = set()
    unnamed = 0
    for key, form in citations:
        if form == "person":
            people.add(key)
        elif form == "organization":
            orgs.add(key)
        elif form not in _COUNTABLE:
            # Includes "unknown": a source we failed to type is still a source
            # we cannot count, so it widens the bound rather than vanishing.
            unnamed += 1

    named = len(people) + len(orgs)
    lower = max(named, 1 if unnamed else 0)
    return SourceCount(len(people), len(orgs), unnamed, lower, named + unnamed)


def bound_is_informative(counts: Iterable[SourceCount]) -> Mapping[str, float]:
    """Whether the bounds are narrow enough to compare stories at all.

    A bound whose width exceeds the spread of the thing being measured cannot
    order two stories, however carefully it was derived. Checking this before
    building on the measure is the same gate that stopped the 25-year
    composition trend, where a 0.369-wide bound sat over a 0.145 signal.

    Args:
        counts: Per-story counts.

    Returns:
        ``mean_width``, ``max_width``, ``between_story_sd`` of the midpoints,
        and ``ratio`` of mean width to that standard deviation. A ratio below 1
        means the typical story's uncertainty is smaller than the variation
        between stories, so comparisons survive; above 1 they do not.
    """
    items = list(counts)
    if not items:
        return {"mean_width": 0.0, "max_width": 0.0, "between_story_sd": 0.0, "ratio": 0.0}

    widths = [c.width for c in items]
    midpoints = [(c.lower + c.upper) / 2 for c in items]
    sd = statistics.stdev(midpoints) if len(midpoints) > 1 else 0.0
    mean_width = statistics.fmean(widths)
    return {
        "mean_width": mean_width,
        "max_width": float(max(widths)),
        "between_story_sd": sd,
        "ratio": mean_width / sd if sd else float("inf"),
    }
