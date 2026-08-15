"""Source counts, and the bounds that unnamed citations force."""

from __future__ import annotations

from covered.counts import SourceCount, bound_is_informative, count_sources


def test_named_sources_are_deduplicated() -> None:
    """A story quoting one person five times has cited one source."""
    counts = count_sources(
        [("jake sullivan", "person")] * 5 + [("the white house", "organization")]
    )
    assert (counts.n_person, counts.n_organization) == (1, 1)
    assert counts.lower == counts.upper == 2
    assert counts.width == 0


def test_an_unnamed_citation_opens_the_bound() -> None:
    """"Officials said" could be Sullivan again, or someone new. Both survive."""
    counts = count_sources([("jake sullivan", "person"), ("officials", "unnamed")])
    assert counts.lower == 1, "the unnamed citation may be Sullivan himself"
    assert counts.upper == 2, "or it may be a second person"


def test_a_story_with_only_unnamed_citations_still_has_a_source() -> None:
    """Unidentifiable is not the same as absent: someone told the reporter."""
    counts = count_sources([("sources", "unnamed"), ("officials", "unnamed")])
    assert counts.lower == 1
    assert counts.upper == 2


def test_an_untyped_source_widens_the_bound_rather_than_vanishing() -> None:
    """Failing to type a source must cost something, or the count flatters us."""
    counts = count_sources([("someone", "unknown")])
    assert counts.n_person == counts.n_organization == 0
    assert (counts.lower, counts.upper) == (1, 1)


def test_a_story_with_no_citations_is_zero_not_one() -> None:
    counts = count_sources([])
    assert (counts.lower, counts.upper, counts.named) == (0, 0, 0)


def test_bounds_are_informative_when_narrower_than_the_spread() -> None:
    """The gate that decides whether the measure can order two stories.

    Measured at 0.31 on the 100-article pilot, against the 25-year composition
    trend which failed the same check at a 0.369 width over a 0.145 signal.
    """
    spread = [
        SourceCount(n, 0, 0, n, n) for n in (2, 4, 6, 8, 10, 12)
    ]
    assert bound_is_informative(spread)["ratio"] == 0.0

    vague = [SourceCount(1, 0, 9, 1, 10) for _ in range(6)]
    assert bound_is_informative(vague)["ratio"] == float("inf")


def test_summary_of_an_empty_set_does_not_raise() -> None:
    assert bound_is_informative([])["ratio"] == 0.0
