"""Correlation between a feature and a continuous target, pooled versus within subjects.

Pooling the rows of all subjects mixes two different questions: *do people with a larger feature
also have a larger target?* (between subjects) and *does the feature move with the target inside one
person?* (within subjects). When subjects differ in scale the two can disagree, even in sign
(Simpson's paradox). :func:`correlation_report` returns both, plus a subject-level summary whose
p-value treats each *subject* (not each row) as one observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class CorrStat:
    """A correlation coefficient with its p-value and the method that produced it."""

    r: float
    p: float
    method: str
    n: int


def choose_method(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> str:
    """``"pearson"`` if both variables pass Shapiro-Wilk (p > alpha), else ``"spearman"``."""
    if len(x) < 3:
        return "spearman"
    normal = stats.shapiro(x).pvalue > alpha and stats.shapiro(y).pvalue > alpha
    return "pearson" if normal else "spearman"


def correlate(x: np.ndarray, y: np.ndarray, method: str = "auto") -> CorrStat:
    """Pearson or Spearman correlation; ``"auto"`` picks one with :func:`choose_method`."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return CorrStat(float("nan"), float("nan"), method if method != "auto" else "n/a", len(x))
    chosen = choose_method(x, y) if method == "auto" else method
    if chosen == "pearson":
        res = stats.pearsonr(x, y)
        return CorrStat(float(res.statistic), float(res.pvalue), "pearson", len(x))
    if chosen == "spearman":
        sp = stats.spearmanr(x, y)
        return CorrStat(float(sp.statistic), float(sp.pvalue), "spearman", len(x))
    raise ValueError("method must be 'auto', 'pearson' or 'spearman'")


@dataclass
class CorrelationResult:
    """Output of :func:`correlation_report`."""

    pooled: CorrStat  # all rows together (mixes between- and within-subject variation)
    within: CorrStat  # all rows after centring x and y inside each subject
    subject_level: CorrStat  # Fisher-z of the per-subject rho, t-test across subjects
    per_group: pd.DataFrame  # one Spearman rho per subject


def correlation_report(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    *,
    log_x: bool = False,
    min_rows: int = 3,
) -> CorrelationResult:
    """Correlation of ``x`` with ``y`` in four views.

    ``log_x`` takes the log of ``x`` first (for positive, multiplicative features).

    * ``pooled``: every row together.
    * ``within``: ``x`` and ``y`` centred inside each subject, then pooled. Removes between-subject
      offsets. Its p-value still counts every row as independent, so read it as a descriptive
      measure of the within-subject slope, not as a significance test.
    * ``per_group``: Spearman rho of each subject with at least ``min_rows`` rows.
    * ``subject_level``: Fisher-z transform of the per-subject rhos and a one-sample t-test
      against 0 across subjects. This is the view whose p-value respects the subject structure.
    """
    data = df[[group, x, y]].dropna().copy()
    xv = data[x].to_numpy(dtype=float)
    if log_x:
        if (xv <= 0).any():
            raise ValueError(f"{x!r} must be strictly positive to take its log")
        data[x] = np.log(xv)
    pooled = correlate(data[x].to_numpy(), data[y].to_numpy())

    centred = data[[x, y]] - data.groupby(group)[[x, y]].transform("mean")
    within = correlate(centred[x].to_numpy(), centred[y].to_numpy())

    rows = []
    for g, d in data.groupby(group, sort=False):
        if len(d) >= min_rows:
            stat = correlate(d[x].to_numpy(), d[y].to_numpy(), method="spearman")
            rows.append({"group": g, "n": len(d), "rho": stat.r, "p": stat.p})
        else:
            rows.append({"group": g, "n": len(d), "rho": np.nan, "p": np.nan})
    per_group = pd.DataFrame(rows)

    rho = per_group["rho"].dropna().to_numpy(dtype=float)
    if len(rho) >= 3:
        z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
        t = stats.ttest_1samp(z, 0.0)
        subject_level = CorrStat(
            float(np.tanh(z.mean())), float(t.pvalue), "fisher-z t-test", len(rho)
        )
    else:
        subject_level = CorrStat(float("nan"), float("nan"), "fisher-z t-test", len(rho))
    return CorrelationResult(
        pooled=pooled, within=within, subject_level=subject_level, per_group=per_group
    )
