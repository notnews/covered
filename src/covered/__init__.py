"""covered: source-concentration analysis of CNN transcripts (2000-2025).

Two separately-reported measures of "who gets quoted":
  (a) on-air speakers  -- from ``NAME, ROLE:`` speaker labels
  (b) cited sources    -- third-party individuals quoted/paraphrased in body text

Headline output: an annual Herfindahl-Hirschman Index (HHI) time series.
"""

from importlib.metadata import version as _version

__version__ = _version("covered")
