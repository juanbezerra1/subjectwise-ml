"""Pre-flight audit of a subject-structured table.

Small-cohort studies fail quietly: a recording copied under a new subject id, a subject that never
performed one of the classes, a class that is three times more common than the others. Each of
these makes a leave-one-subject-out number look better (or worse) than it is. :func:`audit` turns
those mistakes into explicit, named warnings *before* any model is trained.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    """One finding of :func:`audit`."""

    severity: Severity
    code: str
    message: str


@dataclass
class AuditReport:
    """Result of :func:`audit`."""

    n_rows: int
    n_groups: int
    classes: list[str]
    counts: pd.DataFrame  # rows: group, columns: class, values: number of rows
    chance_rate: float  # 1 / number of classes
    majority_rate: float  # accuracy of always predicting the most common class
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """``True`` when there is no error (warnings are allowed)."""
        return not self.errors

    def format(self) -> str:
        """Human-readable report."""
        lines = [
            f"{self.n_rows} rows | {self.n_groups} groups | {len(self.classes)} classes",
            f"chance = {self.chance_rate:.1%} (1/K), "
            f"majority-class rate = {self.majority_rate:.1%}",
            "",
            self.counts.to_string(),
            "",
        ]
        if not self.issues:
            lines.append("no issues found")
        for issue in self.issues:
            lines.append(f"[{issue.severity.upper()}] {issue.code}: {issue.message}")
        return "\n".join(lines)


def _signature(block: np.ndarray) -> str:
    """Order-independent fingerprint of a block of rows (rounded to 6 decimals)."""
    order = np.lexsort(block.T[::-1])
    rounded = np.round(block[order], 6) + 0.0  # + 0.0 turns -0.0 into 0.0
    return hashlib.sha1(rounded.tobytes()).hexdigest()


def audit(
    df: pd.DataFrame,
    features: Sequence[str],
    label: str,
    group: str,
    *,
    min_rows_per_group: int = 3,
) -> AuditReport:
    """Inspect ``df`` for problems that bias a leave-one-subject-out evaluation."""
    cols = list(features)
    issues: list[Issue] = []
    x = df[cols].to_numpy(dtype=float)
    groups = df[group].to_numpy()

    counts = pd.crosstab(df[group], df[label])
    classes = [str(c) for c in counts.columns]
    n_groups = len(counts)
    class_totals = counts.sum(axis=0)
    chance = 1.0 / max(len(classes), 1)
    majority = float(class_totals.max() / class_totals.sum()) if len(df) else float("nan")

    bad = ~np.isfinite(x)
    if bad.any():
        where = [c for c, flag in zip(cols, bad.any(axis=0), strict=True) if flag]
        issues.append(
            Issue(
                "error",
                "non_finite",
                f"{int(bad.any(axis=1).sum())} rows have NaN/inf in {where}; "
                "impute or drop them before modelling",
            )
        )
    if n_groups < 2:
        issues.append(Issue("error", "too_few_groups", "need at least 2 groups for LOSO"))
    elif n_groups == 2:
        issues.append(
            Issue(
                "warning",
                "two_groups",
                "with 2 groups LOSO is a single train/test split in each direction; "
                "the estimate is very noisy",
            )
        )
    if len(classes) < 2:
        issues.append(Issue("error", "single_class", "the label has fewer than 2 classes"))

    tiny = [str(g) for g, n in counts.sum(axis=1).items() if n < min_rows_per_group]
    if tiny:
        issues.append(
            Issue(
                "warning",
                "tiny_group",
                f"groups with fewer than {min_rows_per_group} rows: {tiny}; "
                "within-group normalisation is unreliable for them",
            )
        )
    for g, row in counts.iterrows():
        absent = [str(c) for c, n in row.items() if n == 0]
        if absent:
            issues.append(
                Issue(
                    "warning",
                    "missing_class_in_group",
                    f"group {g!r} has no rows for class(es) {absent}: its normalised features are "
                    "shifted, and its accuracy cannot be compared with the others",
                )
            )

    if majority - chance > 0.05:
        issues.append(
            Issue(
                "warning",
                "imbalanced_classes",
                f"majority-class rate {majority:.1%} is above 1/K = {chance:.1%}: plain accuracy "
                "is inflated; judge the model against the permutation null and balanced accuracy",
            )
        )

    signatures: dict[str, list[str]] = {}
    for g in pd.unique(groups):
        block = x[groups == g]
        if len(block) and np.isfinite(block).all():
            signatures.setdefault(_signature(block), []).append(str(g))
    for twins in (v for v in signatures.values() if len(v) > 1):
        issues.append(
            Issue(
                "warning",
                "duplicate_groups",
                f"groups {twins} have identical feature rows (copied file?). In LOSO the copy of "
                "the held-out subject stays in the training set and the accuracy comes out "
                "optimistic",
            )
        )

    spread = np.nanstd(x, axis=0)
    flat = [c for c, s in zip(cols, spread, strict=True) if s == 0]
    if flat:
        issues.append(Issue("warning", "constant_feature", f"constant features: {flat}"))

    return AuditReport(
        n_rows=len(df),
        n_groups=n_groups,
        classes=classes,
        counts=counts,
        chance_rate=chance,
        majority_rate=majority,
        issues=issues,
    )
