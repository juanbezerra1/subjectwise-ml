"""A toy cohort with a known hierarchical structure, for demos and tests.

Not a physiological model. It mimics the *statistical* shape of small biosignal studies: several
amplitude-like features that grow with a covariate ``x`` (think: produced force) but are multiplied
by a large, subject-specific gain, plus spectral-like features that change a little. Because the
generating process is known, tests can check that estimators recover it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def toy_features(n_amplitude: int = 4, n_spectral: int = 2) -> tuple[list[str], list[str]]:
    """Return ``(features, log_features)`` of :func:`toy_cohort`."""
    amplitude = [f"amp_{j + 1}" for j in range(n_amplitude)]
    spectral = [f"spec_{j + 1}" for j in range(n_spectral)]
    return amplitude + spectral, amplitude


def toy_cohort(
    n_subjects: int = 8,
    levels: Sequence[int] = (5, 20, 35, 50),
    trials_per_level: int = 5,
    *,
    n_amplitude: int = 4,
    n_spectral: int = 2,
    effect: float = 1.0,
    gain_sd: float = 0.5,
    x_sd: float = 2.0,
    x_center: float = 30.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a balanced cohort.

    Generating process, per subject ``s`` and trial with covariate ``x`` (on the log scale for the
    amplitude features)::

        amp_j  = exp(1 + 0.2 j + gain_s + m_s * effect * (0.030 + 0.004 j) * (x - x_center) + noise)
        spec_j = (90 - 8 j) + offset_s - m_s * effect * (0.06 + 0.02 j) * (x - x_center) + noise

    ``gain_s ~ N(0, gain_sd)`` is shared by all amplitude features (the "volume" of that person),
    ``m_s ~ N(1, 0.25)`` scales the slopes, ``offset_s ~ N(0, 4)`` shifts the spectral features.
    ``x`` is the class level plus ``N(0, x_sd)`` jitter. ``effect = 0`` removes every dependence
    on ``x``. Returns ``subject``, ``level``, ``x`` and the feature columns.
    """
    rng = np.random.default_rng(seed)
    features, _ = toy_features(n_amplitude, n_spectral)
    frames = []
    for s in range(1, n_subjects + 1):
        gain = rng.normal(0.0, gain_sd)
        slope_mult = max(rng.normal(1.0, 0.25), 0.2)
        offset = rng.normal(0.0, 4.0)
        level_col = np.repeat(np.asarray(levels), trials_per_level)
        x = level_col + rng.normal(0.0, x_sd, size=len(level_col))
        common = rng.normal(0.0, 0.12, size=len(x))  # shared trial-to-trial fluctuation
        data: dict[str, np.ndarray] = {}
        for j in range(n_amplitude):
            slope = effect * slope_mult * (0.030 + 0.004 * j)
            log_amp = (
                1.0 + 0.2 * j + gain + slope * (x - x_center)
                + common + rng.normal(0.0, 0.18, size=len(x))
            )  # fmt: skip
            data[f"amp_{j + 1}"] = np.exp(log_amp)
        for j in range(n_spectral):
            slope = effect * slope_mult * (0.06 + 0.02 * j)
            data[f"spec_{j + 1}"] = (
                90.0 - 8.0 * j + offset - slope * (x - x_center) + rng.normal(0.0, 3.0, size=len(x))
            )
        frame = pd.DataFrame(data)
        frame.insert(0, "x", x)
        frame.insert(0, "level", level_col)
        frame.insert(0, "subject", f"S{s:02d}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[["subject", "level", "x", *features]]
