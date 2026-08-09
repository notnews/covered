"""Monte Carlo coverage studies for the two intervals in :mod:`covered.metrics`.

``bootstrap_ci`` and ``simulate_hhi_ci`` are the only places in this package that
attach an uncertainty statement to a number that reaches the paper. Until this
file existed neither had been checked against a known truth even once. The three
tests in ``tests/test_metrics.py`` that appear to cover them cannot fail:

* ``test_bootstrap_ci_brackets_mean`` bootstraps ``[0.4] * 100``. Every resample
  of a constant vector is that constant, so both endpoints equal 0.4 by
  construction. A wrong quantile, a wrong ``alpha``, an off-by-one resample, or
  ``return (lo, lo)`` all pass it.
* ``test_simulate_hhi_no_error_is_deterministic`` passes ``recall=1.0,
  precision=1.0``, the one branch in which ``kept = obs`` and ``add_factor = 0``,
  so no simulation runs at all. It tests that ``hhi`` is deterministic.
* ``test_simulate_hhi_with_error_brackets_and_orders`` asserts
  ``lo <= median <= hi``, which holds for any three sorted quantiles of any array
  whatsoever, and ``not isnan(median)``.

Every gate below comes from :mod:`simcheck`, so its tolerance is derived from the
replicate count rather than chosen by hand: ``assert coverage > 0.85`` for a
nominal 95% interval is far too loose at ten thousand replicates and tight enough
to fail spuriously at fifty. Where a one-sided floor is needed --- to show that a
known-bad interval really does under-cover rather than merely score lower ---
:func:`~simcheck.binomial_band` supplies it from that same replicate count.

Four results are worth reading before the tests:

**The percentile bootstrap under-covers badly for a skewed mean.** 0.819 against
a nominal 0.95 on lognormal counts at n=25 over 2000 replicates. This is the
textbook motivation for BCa rather than a defect introduced here, but per-source
quote counts are exactly that shape, so it is pinned rather than assumed away.

**``simulate_hhi_ci`` covers when no source is missed and not when one is.**
0.945 over 1000 replicates on thirty well-covered sources at recall 0.9, against
0.921, 0.892 and 0.873 over 600 as recall falls through 0.9, 0.7 and 0.5 on a
hundred one-off sources and 10, 30 then 50 of them vanish outright. The second is
the weakness ``simulate_hhi_ci``'s own docstring admits, and the size of it is
the surprise: the deficit is real and monotone in the number of sources lost, but
small, because HHI is a share measure dominated by the head and a one-off source
contributes ``(1/N)**2`` to it. Only the recall-0.5 rung is far enough below
nominal for a gate this file can afford to resolve; the other two are recorded.

**``simulate_hhi_ci``'s point correction is arithmetically a no-op.** It
multiplies every source's count by the same factor ``precision / recall`` in
expectation, and HHI is invariant to a common scale, so its median reproduces
``hhi(observed)`` to five decimals at any recall and precision. The module
docstring promises "a corrected series with credible intervals next to the raw
one"; what it delivers is the raw series with a noise band around it. The
interval is real. The correction is not.

**Its coverage is a function of ``n_draws``, and the default is what saves it.**
The interval is a pair of empirical quantiles of ``n_draws`` simulated HHIs, and
``np.nanquantile``'s linear interpolation places the nominal 2.5% endpoint at
about the ``0.025 + 0.975 / n_draws`` quantile of the simulated distribution. The
interval therefore spans about ``0.95 - 1.95 / n_draws`` of it, whatever it is a
quantile of --- 0.948 at the default ``n_draws=1000`` and 0.940 at 199. Checked
directly against a standard normal over 40000 trials: 0.9490 and 0.9413.

That is not a defect of ``simulate_hhi_ci``; it is what a percentile interval
from a finite simulation costs, and at the default it is a fifth of a point. But
it is large enough to swamp a cheap coverage study. Paired over 1000 replicates
of the fixture in ``test_well_covered_sources_cover``, coverage is 0.945 at 1000
draws against 0.938 at 199, with mean interval widths of 0.00443 and 0.00435.
The studies below therefore run at the library default, and this note exists so
that a future reader who reruns them cheaply does not conclude the estimator is
broken.

Runtime is about 55 seconds at the fast tier (100 replicates) and 3 minutes at
the deep tier; both were run before this landed and both pass.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
import pytest
from scipy.stats import norm
from simcheck import (
    Estimate,
    MonteCarloResult,
    assert_coverage,
    assert_unbiased,
    binomial_band,
    monte_carlo,
    reps_for,
)

from covered.hhi import hhi
from covered.metrics import bootstrap_ci, simulate_hhi_ci

# Resamples inside one bootstrap replicate. Small on purpose: the Monte Carlo
# error of a percentile at 399 resamples is far below the coverage differences
# these studies need to resolve, and this runs once per replicate per study.
N_BOOT = 399

# Draws inside one HHI replicate. This one is NOT free to choose: the coverage
# the interval delivers depends on it (see the module docstring), so a study
# that lowers it measures a different interval. 1000 is the library default.
N_DRAWS = 1000

# The cheaper setting, used only where the failure being pinned is a bias several
# times the interval's whole width, so quantile resolution cannot change the
# verdict.
N_DRAWS_COARSE = 199

# Replicate count for the studies that pin a *failure*, fixed rather than taken
# from `reps_for()`. The reason is power, not preference. At the fast tier's 100
# replicates the 3-sigma floor under a nominal 0.95 is 0.884, and the two
# borderline rates pinned below -- 0.819 for the skewed bootstrap and 0.873 for
# the vanishing tail -- sit only 1.7 and 0.3 standard errors under it, so those
# gates would pass or fail on the seed. 600 replicates put the floor at 0.923,
# which is 5.7 and 3.7 standard errors clear of them respectively.
#
# The other two HHI failures do NOT need this and take `reps_for()` instead: they
# measure 0.057 and 0.000, more than twenty standard errors below the floor even
# at 100 replicates, so raising the count would buy nothing but runtime.
PINNED_REPS = 600

Draw = Callable[[np.random.Generator], np.ndarray]


# ---------------------------------------------------------------------------
# Study 1: bootstrap_ci
# ---------------------------------------------------------------------------


def _draw_bernoulli(rng: np.random.Generator, n: int, p: float) -> np.ndarray:
    """One LLM-validation sample: ``n`` adjudicated quotes, correct or not."""
    return rng.binomial(1, p, n).astype(float)


def _draw_poisson(rng: np.random.Generator, n: int, lam: float) -> np.ndarray:
    """Quotes per transcript: a count, right-skewed, but with a light tail."""
    return rng.poisson(lam, n).astype(float)


def _draw_lognormal(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """Quotes per source: the heavy-tailed shape the CNN corpus actually has."""
    return rng.lognormal(0.0, sigma, n)


def _bootstrap_study(
    draw: Draw,
    truth: float,
    *,
    alpha: float = 0.05,
    reps: int | None = None,
    seed: int = 0,
    shrink: float = 1.0,
) -> MonteCarloResult:
    """Run ``bootstrap_ci`` over many fresh samples and record what it did.

    Each replicate passes its own ``seed`` to ``bootstrap_ci``. That is not
    cosmetic: the parameter defaults to 0, so a study that left it alone would
    draw the *same* matrix of resampling positions in every replicate and the
    bootstrap's own noise would be perfectly correlated across them instead of
    averaging out.

    Args:
        draw: Takes a generator, returns one sample.
        truth: The population mean, known because the caller chose the
            distribution.
        alpha: Passed to ``bootstrap_ci``; the interval claims ``1 - alpha``.
        reps: Replicate count; defaults to the current simcheck tier.
        seed: Seed for the replicate stream.
        shrink: Multiply each half-width by this. Only the negative control
            passes anything but 1.0.

    Returns:
        MonteCarloResult: Estimates and coverage flags. ``standard_errors`` stays
        NaN because ``bootstrap_ci`` reports no standard error, so
        ``assert_se_calibrated`` would have nothing to check and says so.
    """
    reps = reps_for() if reps is None else reps

    def replicate(rng: np.random.Generator) -> Estimate:
        sample = draw(rng)
        low, high = bootstrap_ci(
            sample, n_boot=N_BOOT, alpha=alpha, seed=int(rng.integers(2**31))
        )
        if shrink != 1.0:
            middle = 0.5 * (low + high)
            low = middle - shrink * (middle - low)
            high = middle + shrink * (high - middle)
        return Estimate(value=float(np.mean(sample)), lower=low, upper=high)

    return monte_carlo(replicate, truth=truth, reps=reps, seed=seed)


# Measured coverage of the nominal 95% interval over 2000 replicates while this
# file was being written, for reference from the tests below:
#
#     shape                    n=20   n=25   n=40   n=60   n=100  n=200
#     Bernoulli(0.4)              -      -      -  0.961  0.952  0.946
#     Normal(0.4, 0.15)           -  0.928  0.934      -  0.931      -
#     Poisson(3)              0.933      -  0.946      -      -      -
#     lognormal(0, 1.00)      0.842  0.856  0.885      -      -      -
#     lognormal(0, 1.25)      0.805  0.819  0.854      -      -      -
#     lognormal(0, 1.50)      0.755  0.776  0.808      -      -      -
#
# The Normal row is a uniform 1.7-point deficit -- the percentile bootstrap
# applies no t correction -- which is inside the band at both tiers and so is
# recorded here rather than gated.


@pytest.mark.parametrize("n", [60, 200])
def test_a_validation_rate_covers(n: int) -> None:
    """The estimand ``bootstrap_ci`` exists for: an LLM-validation rate.

    The vacuous test this strengthens bootstrapped ``[0.4] * 100``, a vector on
    which every resample is identical, so the interval was a point and the
    assertion held for any implementation. The sample here is Bernoulli(0.4) ---
    the same 0.4, now with sampling variation --- so a wrong quantile moves the
    endpoints and the gate sees it.

    Two sizes because a validation tier of 60 adjudicated quotes and one of 200
    are both plausible here, and the percentile bootstrap is worse at the smaller
    one. Measured 0.961 (n=60) and 0.946 (n=200) over 2000 replicates.
    """
    study = _bootstrap_study(functools.partial(_draw_bernoulli, n=n, p=0.4), truth=0.4)
    assert_unbiased(study, label=f"validation rate, n={n}")
    assert_coverage(study, 0.95, label=f"validation rate, n={n}")


def test_a_count_mean_covers() -> None:
    """A second shape, so the study is not one distribution's accident.

    Quotes per transcript, Poisson(3) at n=40: discrete and right-skewed, but
    light-tailed. Measured 0.946 over 2000 replicates. Read against the lognormal
    rows in the table above, this is the useful contrast: skew alone does not
    break the percentile bootstrap, a heavy tail does.
    """
    study = _bootstrap_study(functools.partial(_draw_poisson, n=40, lam=3.0), truth=3.0)
    assert_unbiased(study, label="quotes per transcript")
    assert_coverage(study, 0.95, label="quotes per transcript")


@pytest.mark.parametrize("n", [100, 200])
def test_alpha_is_honoured(n: int) -> None:
    """``alpha=0.20`` must produce an 80% interval, not a 95% one.

    Together with ``test_a_validation_rate_covers`` this is what an off-by-one
    quantile or a swapped ``alpha / 2`` cannot survive: one fixture, gated at two
    nominal levels, required to hit both. Measured 0.823 (n=100) and 0.810
    (n=200) against a nominal 0.80 over 2000 replicates, against 0.952 and 0.946
    for the same samples at ``alpha=0.05``.
    """
    study = _bootstrap_study(
        functools.partial(_draw_bernoulli, n=n, p=0.4), truth=0.4, alpha=0.20
    )
    assert_coverage(study, 0.80, label=f"alpha=0.20, n={n}")


def test_a_skewed_mean_at_small_n_under_covers() -> None:
    """The finding, pinned because this package points ``bootstrap_ci`` here.

    The percentile bootstrap is not second-order accurate for the mean of a
    skewed distribution: the interval is centred on the sample mean and inherits
    its skew instead of correcting for it. That is the standard motivation for
    BCa and the bootstrap-t, so it is a known property rather than a bug
    introduced here --- but per-source quote counts in the CNN corpus are
    precisely this shape, and a 95% label on an interval that covers 82% of the
    time is a claim the paper should not make silently.

    Measured over 2000 replicates on lognormal(0, 1.25) at n=25: 0.819 against
    a nominal 0.95. The deficit closes only slowly with sample size --- 0.854 at
    n=40 and 0.869 at n=80, both still below the floor. The gate runs 600
    replicates, which puts 0.819 5.7 standard errors under the floor.

    If this ever stops failing, either the generator stopped being heavy-tailed
    or ``bootstrap_ci`` acquired a bias correction. In the second case the fix
    belongs in its docstring, not in a silent deletion of this test.
    """
    study = _bootstrap_study(
        functools.partial(_draw_lognormal, n=25, sigma=1.25),
        truth=float(np.exp(1.25**2 / 2)),
        reps=PINNED_REPS,
    )
    with pytest.raises(AssertionError):
        assert_coverage(study, 0.95, label="lognormal mean, n=25")

    floor, _ = binomial_band(0.95, study.reps)
    assert study.coverage < floor, (
        "the percentile bootstrap should under-cover the mean of a heavy-tailed "
        f"sample at n=25; measured {study.coverage:.3f} against {floor:.3f}"
    )


def test_a_shrunk_interval_is_caught() -> None:
    """A gate that cannot reject a bad interval is certifying nothing.

    The shrink factor is derived, not chosen: ``z(0.90) / z(0.975) = 0.654`` is
    exactly the factor that turns a Gaussian 95% interval into an 80% one. So the
    shrunken interval must fail a 95% gate *and pass an 80% one*, and both halves
    are asserted here. The second half is what separates a gate that measures
    coverage from one that rejects whatever it is handed.

    Measured over 2000 replicates, Bernoulli(0.4) at n=200: 0.790 --- outside
    the band for a nominal 0.95, which at the 600 replicates this gate runs is
    ``[0.923, 0.977]``, and inside it for a nominal 0.80, ``[0.751, 0.849]``.
    """
    shrink = float(norm.ppf(0.90) / norm.ppf(0.975))
    study = _bootstrap_study(
        functools.partial(_draw_bernoulli, n=200, p=0.4),
        truth=0.4,
        reps=PINNED_REPS,
        shrink=shrink,
    )
    with pytest.raises(AssertionError):
        assert_coverage(study, 0.95, label="interval shrunk to 80%")
    assert_coverage(study, 0.80, label="interval shrunk to 80%")


def test_the_two_nominal_levels_are_distinguishable() -> None:
    """Each ``alpha`` study must fail at the *other* study's nominal level.

    Without this, ``test_alpha_is_honoured`` and ``test_a_validation_rate_covers``
    could both be passing on bands wide enough to admit either answer, and an
    implementation that ignored ``alpha`` entirely would satisfy both. Measured
    over 2000 replicates at n=200: 0.946 at ``alpha=0.05`` and 0.810 at
    ``alpha=0.20``, each outside the other's 3-sigma band.
    """
    narrow = _bootstrap_study(
        functools.partial(_draw_bernoulli, n=200, p=0.4),
        truth=0.4,
        alpha=0.20,
        reps=PINNED_REPS,
    )
    wide = _bootstrap_study(
        functools.partial(_draw_bernoulli, n=200, p=0.4),
        truth=0.4,
        alpha=0.05,
        reps=PINNED_REPS,
    )
    with pytest.raises(AssertionError):
        assert_coverage(narrow, 0.95, label="alpha=0.20 judged at 95%")
    with pytest.raises(AssertionError):
        assert_coverage(wide, 0.80, label="alpha=0.05 judged at 80%")


# ---------------------------------------------------------------------------
# Study 2: simulate_hhi_ci
# ---------------------------------------------------------------------------


def _draw_desk_sources(
    rng: np.random.Generator,
    n_sources: int = 30,
    total: int = 6000,
    exponent: float = 1.1,
) -> np.ndarray:
    """A year of quotes over a Zipf-like set of well-covered sources.

    The largest source expects 1724 of the 6000 events and the smallest 41, for
    an expected HHI of 0.122. Every source is large enough that no realistic
    recall makes it vanish --- at recall 0.7 the chance the smallest disappears
    entirely is ``0.3 ** 36``, the observed minimum count averaging 36 --- so
    this is the fixture in which the correction's own model holds exactly.
    """
    ranks = np.arange(1, n_sources + 1, dtype=float)
    weights = ranks**-exponent
    weights /= weights.sum()
    return rng.multinomial(total, weights).astype(float)


def _draw_long_tail_sources(
    rng: np.random.Generator,
    n_top: int = 6,
    n_singleton: int = 100,
    top_total: int = 900,
) -> np.ndarray:
    """Six dominant voices plus a hundred one-off sources.

    The shape ``src/covered/hhi.py`` is written around --- "raw HHI is sensitive
    to the long tail of one-off sources" --- and the fixture in which whole
    sources really do vanish: 10 of the hundred singletons are missed entirely at
    recall 0.9, 30 at 0.7 and 50 at 0.5. The head holds 900 events against the
    tail's 100, for a true HHI of about 0.20.
    """
    ranks = np.arange(1, n_top + 1, dtype=float)
    weights = ranks**-1.0
    weights /= weights.sum()
    top = rng.multinomial(top_total, weights).astype(float)
    return np.concatenate([top, np.ones(n_singleton)])


def _corrupt(
    rng: np.random.Generator,
    true_counts: np.ndarray,
    recall: float,
    precision: float,
    tail_recall: float | None = None,
    tail_cut: float = 5.0,
) -> np.ndarray:
    """Run true counts through the extraction pipeline the docstring describes.

    Drop each true event independently, then add false positives spread over the
    surviving events so the achieved precision equals ``precision`` --- which is
    exactly the model ``simulate_hhi_ci`` inverts when it "removes false
    positives by keeping each observed event with probability ``precision``".

    A negative-binomial allocation of false positives, which is the *exact*
    inverse of that keep-with-probability-p step rather than merely a
    fixed-budget approximation to it, was measured during development and gives
    the same coverage to within Monte Carlo error (0.9417 against 0.9417 at
    recall 0.9 / precision 0.95 over 1200 replicates), so the simpler
    fixed-budget version is used.

    Args:
        rng: Generator.
        true_counts: Events per source before extraction.
        recall: Per-event survival probability for sources above ``tail_cut``,
            and for every source when ``tail_recall`` is None.
        precision: Achieved precision of the observed set.
        tail_recall: A separate, lower survival probability for sources with at
            most ``tail_cut`` true events. None gives the uniform-recall model
            the correction assumes.
        tail_cut: Count at or below which a source counts as tail.

    Returns:
        np.ndarray: Observed counts per source, zero where a source was missed
        entirely.
    """
    if tail_recall is None:
        rates = np.full(true_counts.size, recall)
    else:
        rates = np.where(true_counts <= tail_cut, tail_recall, recall)
    kept = rng.binomial(true_counts.astype(int), rates).astype(float)
    n_true_positive = kept.sum()
    n_false_positive = round(n_true_positive * (1.0 - precision) / precision)
    if n_false_positive > 0:
        kept = kept + rng.multinomial(n_false_positive, kept / n_true_positive).astype(
            float
        )
    return kept


def _hhi_study(
    draw: Draw,
    recall: float,
    precision: float,
    *,
    tail_recall: float | None = None,
    stated_recall: float | None = None,
    stated_precision: float | None = None,
    n_draws: int = N_DRAWS,
    reps: int | None = None,
    seed: int = 0,
) -> MonteCarloResult:
    """Corrupt a known source distribution and ask ``simulate_hhi_ci`` to undo it.

    The truth is redrawn each replicate, so there is no single scalar to hand
    :func:`~simcheck.monte_carlo`. Each replicate therefore reports the *error*,
    ``median - truth``, with both interval endpoints shifted by the same truth,
    against a truth of zero. Shifting all three by one constant leaves the
    coverage indicator exactly unchanged, and it makes ``assert_unbiased`` test
    the correction's bias --- the quantity of interest --- rather than the level
    of HHI, which is a property of the fixture.

    Args:
        draw: Takes a generator, returns true counts per source.
        recall: Recall of the simulated extraction pipeline.
        precision: Precision of the simulated extraction pipeline.
        tail_recall: Lower recall for one-off sources, or None for the uniform
            model the correction assumes.
        stated_recall: The recall handed to ``simulate_hhi_ci``; defaults to the
            true ``recall``.
        stated_precision: Likewise for precision.
        n_draws: Simulation draws inside each call.
        reps: Replicate count; defaults to the current simcheck tier.
        seed: Seed for the replicate stream.

    Returns:
        MonteCarloResult: Signed errors and coverage flags.
    """
    reps = reps_for() if reps is None else reps
    stated_recall = recall if stated_recall is None else stated_recall
    stated_precision = precision if stated_precision is None else stated_precision

    def replicate(rng: np.random.Generator) -> Estimate:
        true_counts = draw(rng)
        truth = hhi(true_counts)
        observed = _corrupt(rng, true_counts, recall, precision, tail_recall)
        result = simulate_hhi_ci(
            {f"s{i}": v for i, v in enumerate(observed) if v > 0},
            recall=stated_recall,
            precision=stated_precision,
            n_draws=n_draws,
            seed=int(rng.integers(2**31)),
        )
        return Estimate(
            value=result["median"] - truth,
            lower=result["lo"] - truth,
            upper=result["hi"] - truth,
        )

    return monte_carlo(replicate, truth=0.0, reps=reps, seed=seed)


@pytest.mark.parametrize(
    ("recall", "precision"), [(0.60, 0.80), (0.80, 0.95), (0.95, 0.70)]
)
def test_the_point_correction_is_a_no_op(recall: float, precision: float) -> None:
    """The mechanism behind every other number in this half of the file.

    ``simulate_hhi_ci`` keeps each observed event with probability ``precision``
    and then adds back ``(1 - recall) / recall`` times what it kept, so in
    expectation every source's count is multiplied by the *same* factor
    ``precision / recall``. HHI is a function of shares and so is invariant to a
    common scale. The corrected median is therefore ``hhi(observed)``, whatever
    recall and precision are set to.

    Measured on a fixed five-source market with raw HHI 0.603340, 4000 draws:

        recall  precision   median      median - raw
        0.60    0.80        0.603394    +0.000055
        0.80    0.95        0.603773    +0.000434
        0.95    0.70        0.603175    -0.000165

    ``covered/metrics.py`` opens by saying missed sources "bias HHI *upward*",
    false positives "bias it *downward*", and that the package reports "a
    corrected series with credible intervals next to the raw one". The interval
    is real; the correction is not. Anything that genuinely moved the point
    estimate --- attributing missed events to *unobserved* sources, or taking a
    per-source recall --- would break this test, and that is the intended way for
    it to break.
    """
    counts = {"a": 500.0, "b": 120.0, "c": 40.0, "d": 3.0, "e": 1.0}
    raw = hhi(list(counts.values()))
    result = simulate_hhi_ci(
        counts, recall=recall, precision=precision, n_draws=4000, seed=1
    )

    assert result["median"] == pytest.approx(raw, abs=5e-3), (
        "the correction rescales every source by the same factor and HHI is "
        "scale-invariant, so the median cannot differ from the raw HHI"
    )
    assert result["lo"] < result["hi"], "the interval must still be non-degenerate"


def test_well_covered_sources_cover() -> None:
    """The generous fixture: the exact model, and no source can vanish.

    Thirty Zipf-distributed sources over 6000 quotes, so the smallest still has
    about 110 events and survives any recall with certainty; recall 0.9,
    precision 0.95, both handed back to ``simulate_hhi_ci`` exactly. This is the
    most favourable test available --- the correction is given the model that
    generated the corruption and the parameters that generated it.

    Measured 0.945 over 1000 replicates at the library's default 1000 draws, with
    a median bias of +0.00003 on an HHI of 0.122 --- which is what
    ``assert_unbiased`` is checking, and which
    ``test_the_point_correction_is_a_no_op`` explains: the median is unbiased
    here because ``hhi(observed)`` is itself unbiased under a uniform recall, not
    because anything was corrected.

    The same 1000 replicates at 199 draws give 0.938, and that seven-tenths of a
    point is the ``n_draws`` artefact described in the module docstring rather
    than a property of the estimator.
    """
    study = _hhi_study(_draw_desk_sources, recall=0.9, precision=0.95)
    assert_unbiased(study, label="HHI correction, well-covered sources")
    assert_coverage(study, 0.95, label="HHI, well-covered sources")


def test_a_vanishing_tail_under_covers() -> None:
    """The weakness ``simulate_hhi_ci``'s own docstring admits, measured.

    "Missed events are conservatively attributed to already-observed sources, so
    the correction is a lower bound on tail diversity." A source missed entirely
    is not in the input at all, so nothing in the simulation can put it back; the
    imputed market has fewer sources than the real one and reads more
    concentrated. The interval sits above the truth and the coverage deficit
    grows with the number of sources lost, monotonically, in exactly the way that
    sentence predicts. A hundred one-off sources, precision 0.9, 600 replicates
    at 199 draws:

        recall   sources lost   coverage   mean signed error
        0.9      10             0.921      +0.00045
        0.7      30             0.892      +0.00073
        0.5      50             0.873      +0.00035

    This is gated at recall 0.5, the only one of the three the runtime budget can
    resolve: 0.873 against a 600-replicate floor of 0.923 is 3.7 standard errors,
    while 0.921 at recall 0.9 would need about 2400 replicates and eighteen
    minutes.

    It is a *small* effect for HHI, and the reason is worth writing down next to
    it, because it is the opposite of what the module docstring implies. Under a
    uniform recall the tail is thinned in the same proportion as the head, so the
    surviving shares stay nearly unbiased even though whole sources are gone ---
    HHI reads shares, is dominated by the head, and a one-off source contributes
    ``(1/N)**2`` to it. Losing half the one-off sources moves the point estimate
    by 0.0004 on an HHI of 0.20. The admitted weakness bites hard on the
    *diversity* measures in ``covered.hhi``, ``n_distinct`` and
    ``normalized_entropy``, which have no interval at all, and only barely on
    HHI. What genuinely breaks the HHI interval is non-uniform recall, which the
    next test measures.
    """
    study = _hhi_study(
        _draw_long_tail_sources,
        recall=0.5,
        precision=0.9,
        n_draws=N_DRAWS_COARSE,
        reps=PINNED_REPS,
    )
    with pytest.raises(AssertionError):
        assert_coverage(study, 0.95, label="HHI, vanishing tail")

    floor, _ = binomial_band(0.95, study.reps)
    assert study.coverage < floor, (
        "sources missed entirely cannot be restored, so the interval should "
        f"under-cover; measured {study.coverage:.3f} against {floor:.3f}"
    )
    assert study.bias > 0.0, (
        "and the residual must be upward -- a market with sources removed reads "
        "more concentrated, not less"
    )


def test_tail_biased_recall_is_caught_under_covering() -> None:
    """The second finding, and the case the module docstring actually claims.

    ``covered/metrics.py`` opens by saying missed sources "drop mostly tail/one-
    off voices and bias HHI *upward*". That is a statement about recall being
    *lower for the tail than for the headline sources*, and it is the realistic
    case: an extractor that reliably attributes a quote to the President will
    miss a one-time caller. But ``simulate_hhi_ci`` takes a single scalar recall
    and applies it to every source alike, and --- per
    ``test_the_point_correction_is_a_no_op`` --- that rescaling cancels out of
    HHI entirely. The correction is structurally incapable of removing the bias
    its own module docstring is written to motivate.

    Here the six dominant sources are recalled at 0.9 and the one-off sources at
    0.6, precision 0.9, and ``simulate_hhi_ci`` is told recall is 0.9. Observed
    shares tilt toward the dominant sources, HHI reads high, and the interval ---
    a noise band around an uncorrected point --- sits entirely above the truth.

    Measured 0.057 over 600 replicates against a nominal 0.95, with a mean signed
    error of +0.0148 against a mean interval width of 0.0156 --- the bias is
    almost the entire width of the interval, so it clears the truth on nineteen
    replicates in twenty. Forty of the hundred one-off sources vanish outright.
    """
    study = _hhi_study(
        _draw_long_tail_sources,
        recall=0.9,
        precision=0.9,
        tail_recall=0.6,
        n_draws=N_DRAWS_COARSE,
    )
    with pytest.raises(AssertionError):
        assert_coverage(study, 0.95, label="HHI, tail-biased recall")

    floor, _ = binomial_band(0.95, study.reps)
    assert study.coverage < floor, (
        "a scalar recall cannot correct a tail-biased one; measured "
        f"{study.coverage:.3f} against a floor of {floor:.3f}"
    )
    assert float(np.mean(study.estimates)) > 0.0, (
        "and the residual bias must be upward, as the module docstring says"
    )


def test_assuming_perfect_extraction_is_caught() -> None:
    """The negative control, and it reuses the existing test's own fixture.

    ``test_simulate_hhi_no_error_is_deterministic`` calls
    ``simulate_hhi_ci(recall=1.0, precision=1.0)`` and asserts the three returned
    quantiles are equal. They are, because that branch skips the simulation
    entirely, so the test proves nothing about the interval --- but the same call
    makes a perfectly good *broken estimator* to point a coverage gate at.

    Given genuinely corrupted counts (recall 0.8, precision 0.9), claiming
    perfect extraction returns a degenerate interval at ``hhi(observed)``, which
    contains the truth only when the two coincide to the last decimal. Measured
    0.000 over 600 replicates, mean width exactly 0.0. If this ever stops
    failing, ``assert_coverage`` has stopped measuring coverage and every other
    gate here is certifying nothing.
    """
    study = _hhi_study(
        _draw_desk_sources,
        recall=0.8,
        precision=0.9,
        stated_recall=1.0,
        stated_precision=1.0,
        n_draws=N_DRAWS_COARSE,
    )
    with pytest.raises(AssertionError):
        assert_coverage(study, 0.95, label="HHI assuming perfect extraction")

    floor, _ = binomial_band(0.95, study.reps)
    assert study.coverage < floor, (
        "a degenerate interval cannot cover; measured "
        f"{study.coverage:.3f} against a floor of {floor:.3f}"
    )
