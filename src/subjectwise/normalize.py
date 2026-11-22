"""Within-subject normalisation.

Between-subject differences in *scale* (electrode gain, skin impedance, how hard a person squeezes,
...) often dominate the feature space of a small cohort. Standardising every feature inside each
subject removes that nuisance and leaves the variation that is shared across people.

The statistics are computed from the subject's **own rows and never from the labels**, so a model
evaluated leave-one-subject-out stays leakage-free. The price is transductive: at deployment a new
user needs a short unlabelled calibration block that covers the range of the task, otherwise the
subject's mean and spread are biased.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def within_group_zscore(
    df: pd.DataFrame,
    features: Sequence[str],
    group: str,
    *,
    log_features: Sequence[str] = (),
    ddof: int = 1,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Z-score every feature inside each group (subject). Returns a modified copy.

    Parameters
    ----------
    log_features:
        Strictly positive, multiplicative features (RMS, power, ratios, ...) that are put on a log
        scale *before* the z-score. Zeros are clipped to ``eps``; negative values raise.
    ddof:
        Delta degrees of freedom of the per-group standard deviation (1 = sample SD).

    Zero-variance columns, and groups with a single row, become 0. NaN stays NaN.
    """
    cols = list(features)
    log_set = set(log_features)
    unknown = log_set - set(cols)
    if unknown:
        raise ValueError(f"log_features not in features: {sorted(unknown)}")
    if df[group].isna().any():
        raise ValueError(f"group column {group!r} contains missing values")

    x = df[cols].to_numpy(dtype=float, copy=True)
    for j, name in enumerate(cols):
        if name in log_set:
            if np.nanmin(x[:, j]) < 0:
                raise ValueError(f"feature {name!r} has negative values; it cannot be log-scaled")
            x[:, j] = np.log(np.clip(x[:, j], eps, None))

    grouped = pd.DataFrame(x, columns=cols).groupby(df[group].to_numpy(), sort=False)
    mean = grouped.transform("mean").to_numpy(dtype=float)
    std = grouped.transform("std", ddof=ddof).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 0, (x - mean) / std, 0.0)
    z[np.isnan(x)] = np.nan

    out = df.copy()
    out[cols] = z
    return out
