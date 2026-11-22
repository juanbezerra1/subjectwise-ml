"""Cross-validation that respects the subject structure of the data.

Rows from the same subject are not independent: they share electrode placement, anatomy, effort and
many other nuisance factors. A random split puts rows of every subject on both sides of the
train/test boundary, the model "recognises the person", and the reported accuracy is optimistic.
Holding out whole subjects (leave-one-subject-out, LOSO) measures what matters for a *new* person.

The headline number is the **mean of the per-subject accuracies** (every subject counts once,
however many rows it has) together with its standard deviation across subjects. The pooled balanced
accuracy is reported next to it because classes are rarely perfectly balanced.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold

from .models import make_model
from .normalize import within_group_zscore

Scheme = Literal["loso", "group_kfold", "kfold"]
Normalize = Literal["none", "within_group"]
DEFAULT_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("kfold", "none"),  # naive random split: leaks subject identity
    ("loso", "none"),  # honest estimate
    ("loso", "within_group"),  # honest estimate + per-subject normalisation
)


def prepare(
    df: pd.DataFrame,
    features: Sequence[str],
    label: str,
    group: str,
    *,
    normalize: str = "none",
    log_features: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, y, groups)`` arrays, optionally within-group normalised.

    The normalisation never looks at ``label``, so it can be applied once to the whole table and
    then reused by every cross-validation fold and every permutation.
    """
    cols = list(features)
    if normalize == "within_group":
        data = within_group_zscore(df, cols, group, log_features=log_features)
    elif normalize == "none":
        data = df
    else:
        raise ValueError("normalize must be 'none' or 'within_group'")
    x = data[cols].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("features contain NaN/inf; impute or drop those rows first")
    return x, _class_array(data[label]), data[group].to_numpy(dtype=object)


def _class_array(values: pd.Series) -> np.ndarray:
    """Class labels as a numpy array that scikit-learn recognises as a classification target.

    Text (or mixed) labels become strings; integers and booleans are kept; whole-number floats are
    kept; anything continuous is rejected, because it is not a set of classes.
    """
    if values.isna().any():
        raise ValueError("the label column contains missing values")
    y = values.to_numpy()
    if y.dtype == object:
        return y.astype(str)
    if y.dtype.kind == "f" and not np.all(np.mod(y, 1) == 0):
        raise ValueError("labels must be discrete classes, not continuous values")
    return y


def predict_loso(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, model: str = "lda", random_state: int = 0
) -> np.ndarray:
    """Leave-one-group-out predictions (one model fitted per held-out group)."""
    pred = np.empty(len(y), dtype=y.dtype)
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        pipe = make_model(model, random_state).fit(x[train], y[train])
        pred[test] = pipe.predict(x[test])
    return pred


def group_accuracies(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Accuracy of every group, in the order of ``np.unique(groups)``."""
    _, codes = np.unique(groups.astype(str), return_inverse=True)
    hits = np.bincount(codes, weights=(y_true == y_pred).astype(float))
    return hits / np.bincount(codes)


def _ordered_labels(*arrays: np.ndarray) -> list[object]:
    """Distinct values of the arrays in their natural order (numeric labels stay numeric)."""
    values: set[object] = set()
    for a in arrays:
        values.update(a.tolist())
    try:
        return sorted(values)  # type: ignore[type-var]
    except TypeError:
        return sorted(values, key=str)


def _balanced(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    with warnings.catch_warnings():
        # A held-out subject can lack a class that the model predicts; that is expected in LOSO.
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return float(balanced_accuracy_score(y_true, y_pred))


@dataclass
class EvalResult:
    """Predictions of one cross-validation run plus the usual summaries."""

    scheme: str
    model: str
    normalize: str
    y_true: np.ndarray
    y_pred: np.ndarray
    groups: np.ndarray

    @property
    def labels(self) -> list[str]:
        return [str(v) for v in _ordered_labels(self.y_true, self.y_pred)]

    def per_group(self) -> pd.DataFrame:
        """One row per group: rows, accuracy and balanced accuracy."""
        rows = []
        for g in pd.unique(self.groups):
            m = self.groups == g
            rows.append(
                {
                    "group": g,
                    "n": int(m.sum()),
                    "accuracy": float(accuracy_score(self.y_true[m], self.y_pred[m])),
                    "balanced_accuracy": _balanced(self.y_true[m], self.y_pred[m]),
                }
            )
        return pd.DataFrame(rows)

    @property
    def mean_accuracy(self) -> float:
        """Mean over groups of the per-group accuracy (every subject counts once)."""
        return float(group_accuracies(self.y_true, self.y_pred, self.groups).mean())

    @property
    def sd_accuracy(self) -> float:
        """Sample standard deviation of the per-group accuracies (0 with a single group)."""
        acc = group_accuracies(self.y_true, self.y_pred, self.groups)
        return float(acc.std(ddof=1)) if len(acc) > 1 else 0.0

    @property
    def pooled_accuracy(self) -> float:
        return float(accuracy_score(self.y_true, self.y_pred))

    @property
    def balanced_accuracy(self) -> float:
        """Pooled balanced accuracy (mean recall over classes)."""
        return _balanced(self.y_true, self.y_pred)

    @property
    def f1_macro(self) -> float:
        return float(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0))

    def confusion(self) -> pd.DataFrame:
        """Pooled confusion matrix (rows: true label, columns: predicted label)."""
        native = _ordered_labels(self.y_true, self.y_pred)
        names = [str(v) for v in native]
        cm = confusion_matrix(self.y_true, self.y_pred, labels=native)
        return pd.DataFrame(cm, index=pd.Index(names, name="true"), columns=names)

    def summary(self) -> dict[str, float]:
        return {
            "mean_accuracy": self.mean_accuracy,
            "sd_accuracy": self.sd_accuracy,
            "pooled_accuracy": self.pooled_accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "n_groups": float(len(np.unique(self.groups.astype(str)))),
        }


def _splits(
    scheme: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    n_groups = len(set(groups.tolist()))
    if scheme in ("loso", "group_kfold") and n_groups < 2:
        raise ValueError(f"{scheme} needs at least 2 groups")
    if scheme == "loso":
        return LeaveOneGroupOut().split(x, y, groups)
    if scheme == "group_kfold":
        return GroupKFold(n_splits=min(n_splits, n_groups)).split(x, y, groups)
    if scheme == "kfold":
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(x, y)
    raise ValueError("scheme must be 'loso', 'group_kfold' or 'kfold'")


def evaluate(
    df: pd.DataFrame,
    features: Sequence[str],
    label: str,
    group: str,
    *,
    model: str = "lda",
    scheme: Scheme = "loso",
    normalize: Normalize = "none",
    log_features: Sequence[str] = (),
    n_splits: int = 5,
    random_state: int = 0,
) -> EvalResult:
    """Cross-validate ``model`` on a table whose rows belong to groups (subjects).

    Schemes
    -------
    ``"loso"``
        Every fold holds out one whole subject.
    ``"group_kfold"``
        ``n_splits`` folds; a subject is never split across train and test.
    ``"kfold"``
        Shuffled stratified k-fold that *ignores* the subjects. It exists only to quantify how
        optimistic naive validation is; do not report it as performance.

    For the grouped schemes the function checks that no group lands on both sides of a split.
    """
    x, y, groups = prepare(
        df, features, label, group, normalize=normalize, log_features=log_features
    )
    pred = np.empty(len(y), dtype=y.dtype)
    for train, test in _splits(scheme, x, y, groups, n_splits, random_state):
        if scheme != "kfold":
            shared = set(groups[train].tolist()) & set(groups[test].tolist())
            if shared:
                raise AssertionError(f"group leakage between train and test: {sorted(shared)}")
        pipe = make_model(model, random_state).fit(x[train], y[train])
        pred[test] = pipe.predict(x[test])
    return EvalResult(
        scheme=scheme, model=model, normalize=normalize, y_true=y, y_pred=pred, groups=groups
    )


def compare_schemes(
    df: pd.DataFrame,
    features: Sequence[str],
    label: str,
    group: str,
    *,
    models: Sequence[str] = ("lda", "svm_rbf"),
    scenarios: Sequence[tuple[str, str]] = DEFAULT_SCENARIOS,
    log_features: Sequence[str] = (),
    n_splits: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Run every model under every ``(scheme, normalize)`` scenario; one tidy row each."""
    rows = []
    for scheme, normalize in scenarios:
        for name in models:
            result = evaluate(
                df,
                features,
                label,
                group,
                model=name,
                scheme=scheme,  # type: ignore[arg-type]
                normalize=normalize,  # type: ignore[arg-type]
                log_features=log_features,
                n_splits=n_splits,
                random_state=random_state,
            )
            rows.append(
                {"model": name, "scheme": scheme, "normalize": normalize, **result.summary()}
            )
    return pd.DataFrame(rows)


def leakage_gap(table: pd.DataFrame) -> pd.DataFrame:
    """Optimism of the random split, per model, in accuracy points.

    Needs the ``("kfold", "none")`` and ``("loso", "none")`` rows of :func:`compare_schemes`.
    """
    naive = table[(table["scheme"] == "kfold") & (table["normalize"] == "none")]
    honest = table[(table["scheme"] == "loso") & (table["normalize"] == "none")]
    merged = naive.merge(honest, on="model", suffixes=("_kfold", "_loso"))
    return pd.DataFrame(
        {
            "model": merged["model"],
            "kfold_accuracy": merged["mean_accuracy_kfold"],
            "loso_accuracy": merged["mean_accuracy_loso"],
            "gap_points": (merged["mean_accuracy_kfold"] - merged["mean_accuracy_loso"]) * 100,
        }
    ).reset_index(drop=True)
