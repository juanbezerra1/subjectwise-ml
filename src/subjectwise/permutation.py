"""Permutation test for the chance level of a leave-one-subject-out classifier.

"Chance = 1/K" is only true for a perfectly balanced design and an uninformative classifier. With
unbalanced classes a model that always answers the most common class already beats 1/K. The honest
reference is a *null distribution*: shuffle the labels, run the very same LOSO pipeline, and see how
often chance alone reaches the observed score.

The labels are shuffled **inside every subject**. That keeps each subject's class composition
(and therefore the majority-class baseline) intact and only destroys the link between features and
labels, which is exactly the null hypothesis of interest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .cv import _balanced, group_accuracies, predict_loso, prepare
from .tuning import Grid, Tuning, make_tuning

Metric = Literal["mean_accuracy", "balanced_accuracy"]


def permute_within_groups(
    y: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle ``y`` independently inside each group."""
    out = y.copy()
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = y[rng.permutation(idx)]
    return out


def _score(metric: str, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    if metric == "mean_accuracy":
        return float(group_accuracies(y_true, y_pred, groups).mean())
    return _balanced(y_true, y_pred)


@dataclass
class PermutationResult:
    """Observed score and its permutation null."""

    metric: str
    observed: float
    null: np.ndarray
    tuned: bool = False  # the null repeats the tuning of the observed score

    @property
    def n_perm(self) -> int:
        return len(self.null)

    @property
    def p_value(self) -> float:
        """``(1 + #{null >= observed}) / (1 + n_perm)``: never exactly zero."""
        return float((1 + np.sum(self.null >= self.observed)) / (1 + self.n_perm))

    @property
    def null_mean(self) -> float:
        return float(self.null.mean())

    @property
    def null_p95(self) -> float:
        return float(np.percentile(self.null, 95))

    def summary(self) -> dict[str, float]:
        return {
            "observed": self.observed,
            "null_mean": self.null_mean,
            "null_p95": self.null_p95,
            "p_value": self.p_value,
            "n_perm": float(self.n_perm),
        }


def _one_permutation(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: str,
    metric: str,
    seed: int,
    tuning: Tuning | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    y_perm = permute_within_groups(y, groups, rng)
    pred = predict_loso(x, y_perm, groups, model=model, tuning=tuning)
    return _score(metric, y_perm, pred, groups)


def permutation_test(
    df: pd.DataFrame,
    features: Sequence[str],
    label: str,
    group: str,
    *,
    model: str = "lda",
    normalize: str = "none",
    log_features: Sequence[str] = (),
    metric: Metric = "mean_accuracy",
    n_perm: int = 500,
    seed: int = 0,
    n_jobs: int = 1,
    tune: bool = False,
    grid: Grid | None = None,
    inner_scheme: Literal["loso", "group_kfold"] = "loso",
    inner_splits: int = 5,
) -> PermutationResult:
    """Leave-one-subject-out score and its within-subject permutation null.

    The normalisation never uses labels, so it is applied once; only the labels are shuffled.
    Each permutation draws from its own child seed, which makes the result independent of
    ``n_jobs``. With ``n_perm`` permutations the smallest attainable p-value is ``1/(n_perm+1)``.

    With ``tune=True`` (see :func:`subjectwise.evaluate`) the observed score comes from the nested
    procedure and **every permutation repeats the tuning** on its shuffled labels, so the null
    belongs to the procedure that was actually run. That multiplies the work by the size of the grid
    and the number of training subjects: use few permutations, ``n_jobs`` or a small ``grid``.
    """
    if metric not in ("mean_accuracy", "balanced_accuracy"):
        raise ValueError("metric must be 'mean_accuracy' or 'balanced_accuracy'")
    tuning = make_tuning(
        model, tune=tune, grid=grid, inner_scheme=inner_scheme, inner_splits=inner_splits
    )
    x, y, groups = prepare(
        df, features, label, group, normalize=normalize, log_features=log_features
    )
    if tuning is not None and len(set(groups.tolist())) < 3:
        raise ValueError("tuning needs at least 3 subjects")
    observed = _score(metric, y, predict_loso(x, y, groups, model=model, tuning=tuning), groups)
    seeds = np.random.default_rng(seed).integers(0, 2**32 - 1, size=n_perm)
    null = Parallel(n_jobs=n_jobs)(
        delayed(_one_permutation)(x, y, groups, model, metric, int(s), tuning) for s in seeds
    )
    return PermutationResult(
        metric=metric,
        observed=observed,
        null=np.asarray(null, dtype=float),
        tuned=tuning is not None,
    )
