"""Sensitivity of the result to individual subjects.

With five or six subjects, a single odd participant can move the headline number a lot.
:func:`leave_group_out` re-runs the *same* pipeline on the same pre-computed table once with
everybody and once per excluded subject, so the only thing that changes between rows is the cohort.

This is a diagnostic, not a licence to drop subjects. Exclusion criteria must be fixed **before**
looking at the accuracy; choosing whom to drop after seeing the result inflates the estimate.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .cv import evaluate


def leave_group_out(
    df: pd.DataFrame,
    features: Sequence[str],
    label: str,
    group: str,
    *,
    model: str = "lda",
    normalize: str = "within_group",
    log_features: Sequence[str] = (),
    groups: Sequence[object] | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """One row for the full cohort and one row per excluded group.

    ``delta_points`` is the change of the mean per-subject accuracy, in accuracy points, relative to
    the full cohort. ``groups`` restricts which subjects are dropped (default: all of them).
    """
    everyone = list(pd.unique(df[group]))
    targets = everyone if groups is None else list(groups)

    def run(table: pd.DataFrame) -> dict[str, float]:
        result = evaluate(
            table,
            features,
            label,
            group,
            model=model,
            scheme="loso",
            normalize=normalize,  # type: ignore[arg-type]
            log_features=log_features,
            random_state=random_state,
        )
        return result.summary()

    base = run(df)
    rows = [
        {
            "scenario": "all",
            "dropped": "",
            "n_groups": len(everyone),
            "n_rows": len(df),
            "mean_accuracy": base["mean_accuracy"],
            "sd_accuracy": base["sd_accuracy"],
            "balanced_accuracy": base["balanced_accuracy"],
            "delta_points": 0.0,
        }
    ]
    for g in targets:
        sub = df[df[group] != g]
        summary = run(sub)
        rows.append(
            {
                "scenario": f"without {g}",
                "dropped": str(g),
                "n_groups": len(everyone) - 1,
                "n_rows": len(sub),
                "mean_accuracy": summary["mean_accuracy"],
                "sd_accuracy": summary["sd_accuracy"],
                "balanced_accuracy": summary["balanced_accuracy"],
                "delta_points": (summary["mean_accuracy"] - base["mean_accuracy"]) * 100,
            }
        )
    return pd.DataFrame(rows)
