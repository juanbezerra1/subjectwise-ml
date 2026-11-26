"""Choosing hyper-parameters without ever looking at the subject that is being predicted.

Tuning ``C`` or ``gamma`` on the folds that report the accuracy makes that accuracy optimistic: the
settings were picked because they did well on those very subjects. Tuning with a random inner split
is worse, because it brings back the subject leakage that leave-one-subject-out was there to remove.

The remedy is **nested** validation. For every outer fold the held-out subject (or subjects) is
set aside first; the settings are then chosen by an inner cross-validation that is grouped by
subject as well and runs on the training subjects only; the model refitted with the chosen settings
predicts the held-out subject once. The inner score of the chosen settings is kept next to the outer
accuracy: their difference is the optimism that choosing the best of several candidates adds.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .models import make_model

InnerScheme = Literal["loso", "group_kfold"]
Grid = Mapping[str, Sequence[object]]

#: Models that have a default grid. Any other model can be tuned with an explicit ``grid``.
TUNABLE_MODELS = ("svm_rbf", "svm_linear", "rf")

#: Two scores closer than this are a tie (the earliest candidate wins), not a difference.
_TIE = 1e-12


def default_grid(model: str, n_features: int) -> dict[str, tuple[object, ...]]:
    """The candidates tried for ``model``; the first value of every list is the fixed default.

    Ties go to the earliest candidate, so the default settings are kept unless something else is
    clearly better on the training subjects. ``gamma`` is written out in numbers: after the
    standardisation every model starts with, scikit-learn's ``"scale"`` is ``1 / n_features``.
    """
    if model == "svm_rbf":
        scale = 1.0 / n_features
        return {"C": (1.0, 0.1, 10.0, 100.0), "gamma": (scale, scale / 10, scale * 10)}
    if model == "svm_linear":
        return {"C": (1.0, 0.01, 0.1, 10.0)}
    if model == "rf":
        return {"min_samples_leaf": (1, 3, 8), "max_features": ("sqrt", 0.5)}
    raise ValueError(
        f"no default grid for {model!r}; tunable models are {TUNABLE_MODELS}, or pass grid="
    )


@dataclass(frozen=True)
class Tuning:
    """How the settings are chosen inside the training subjects."""

    grid: Grid | None = None  # None: the default grid of the model
    inner: InnerScheme = "loso"
    inner_splits: int = 5  # only for ``inner="group_kfold"``


def make_tuning(
    model: str,
    *,
    tune: bool,
    grid: Grid | None = None,
    inner_scheme: str = "loso",
    inner_splits: int = 5,
) -> Tuning | None:
    """Validate the tuning options; ``None`` when ``tune`` is off."""
    if not tune:
        if grid is not None:
            raise ValueError("grid is only used with tune=True")
        return None
    if inner_scheme not in ("loso", "group_kfold"):
        raise ValueError("inner_scheme must be 'loso' or 'group_kfold'")
    if inner_splits < 2:
        raise ValueError("inner_splits must be at least 2")
    if grid is None:
        default_grid(model, 1)  # raises for a model that has none
    else:
        _check_grid(grid)
    return Tuning(grid=grid, inner=inner_scheme, inner_splits=inner_splits)  # type: ignore[arg-type]


def _check_grid(grid: Grid) -> None:
    if not grid:
        raise ValueError("grid must name at least one parameter")
    for name, values in grid.items():
        if isinstance(values, str) or not len(values):
            raise ValueError(f"grid[{name!r}] must be a non-empty list of values")


def resolve_grid(model: str, n_features: int, grid: Grid | None) -> dict[str, tuple[object, ...]]:
    """The grid to use: the one given, or the default of ``model``."""
    if grid is None:
        return default_grid(model, n_features)
    _check_grid(grid)
    return {name: tuple(values) for name, values in grid.items()}


def candidates(grid: Grid) -> list[dict[str, object]]:
    """Every combination of the grid, the first value of each parameter varying slowest."""
    names = list(grid)
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*grid.values())]


def mean_group_accuracy(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    """Mean over groups of the per-group accuracy: every subject counts once."""
    _, codes = np.unique(groups.astype(str), return_inverse=True)
    hits = np.bincount(codes, weights=(y_true == y_pred).astype(float))
    return float((hits / np.bincount(codes)).mean())


@dataclass
class Choice:
    """The settings chosen on one set of training subjects."""

    params: dict[str, object]
    score: float  # inner mean per-subject accuracy of the chosen settings
    table: pd.DataFrame = field(repr=False)  # every candidate and its inner score


def _inner_splits(
    scheme: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        raise ValueError("tuning needs at least 2 training subjects")
    if scheme == "loso":
        return list(LeaveOneGroupOut().split(x, y, groups))
    return list(GroupKFold(n_splits=min(n_splits, n_groups)).split(x, y, groups))


def select_params(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: str,
    grid: Grid,
    *,
    scheme: InnerScheme = "loso",
    n_splits: int = 5,
    random_state: int = 0,
) -> Choice:
    """Best candidate of ``grid`` by grouped cross-validation on ``(x, y, groups)``.

    The arrays must hold the **training subjects only**: whatever is passed here is what the choice
    is allowed to see. Every candidate is scored by the mean over subjects of the accuracy it gets
    on the subjects (or folds of subjects) it did not train on; ties go to the earliest candidate.
    """
    options = candidates(grid)
    splits = _inner_splits(scheme, x, y, groups, n_splits)
    predicted = [np.empty(len(y), dtype=y.dtype) for _ in options]
    for train, test in splits:
        shared = set(groups[train].tolist()) & set(groups[test].tolist())
        if shared:
            raise AssertionError(f"group leakage inside the tuning loop: {sorted(shared)}")
        for k, params in enumerate(options):
            fitted = make_model(model, random_state, params).fit(x[train], y[train])
            predicted[k][test] = fitted.predict(x[test])
    scores = [mean_group_accuracy(y, p, groups) for p in predicted]
    best = next(k for k, s in enumerate(scores) if s >= max(scores) - _TIE)
    table = pd.DataFrame(options).assign(inner_score=scores)
    return Choice(params=options[best], score=scores[best], table=table)


def fit_predict_tuned(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    x_test: np.ndarray,
    model: str,
    tuning: Tuning,
    random_state: int = 0,
) -> tuple[np.ndarray, Choice]:
    """Choose the settings on the training subjects, refit with them, predict ``x_test``.

    ``x_test`` is only used by the final ``predict``; nothing before it can see it.
    """
    grid = resolve_grid(model, x_train.shape[1], tuning.grid)
    choice = select_params(
        x_train, y_train, groups_train, model, grid,
        scheme=tuning.inner, n_splits=tuning.inner_splits, random_state=random_state,
    )  # fmt: skip
    final = make_model(model, random_state, choice.params).fit(x_train, y_train)
    return final.predict(x_test), choice
