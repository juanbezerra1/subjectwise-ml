"""A small hierarchical generator of synthetic subjects, fitted on pilot data.

Model (per feature ``j``, on a log scale for multiplicative features)::

    y[i, j] = a[s, j] + b[s, j] * (x[i] - x0) + e[i, j]        # subject s, trial i

* ``x`` is a continuous covariate (for example the produced force); ``x0`` centres it so that the
  intercept and slope estimates are decoupled.
* ``(a[s], b[s])`` are fitted by least squares for every pilot subject, and a *new* subject is drawn
  as ``mu + w @ D`` with ``w ~ N(0, s2 / (k - 1))``: a Gaussian with the empirical between-subject
  covariance of the ``k`` pilot subjects, which keeps the correlation between intercepts and slopes
  of different features.
* ``e`` is multivariate Gaussian with a residual covariance shared by all subjects. The correlation
  matrix is estimated with Ledoit-Wolf shrinkage; it is shrunk *on standardised residuals*, because
  shrinking raw covariances towards a scaled identity would drag features of very different scales
  (log-RMS ~ 0.3, a slope ~ 4, a frequency ~ 3 Hz) towards a common variance.
* With ``variance_uncertainty`` the between-subject variance itself is drawn from its sampling
  distribution (``s2 = (k - 1) / chi2(k - 1)``). With few pilot subjects this is what makes the
  uncertainty of the pilot show up in the simulated studies instead of hiding in them.

Known limitations (read before trusting a power curve): the per-subject estimates carry sampling
noise, which inflates the estimated between-subject spread (the generator is conservative); the
model is Gaussian and linear in ``x``; and it can only reproduce what the pilot subjects show.
:func:`calibration_check` measures how well it does on subjects it has not seen.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf

_TINY = 1e-12


def _to_model_scale(
    values: np.ndarray, columns: Sequence[str], log_set: frozenset[str]
) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    for j, name in enumerate(columns):
        if name in log_set:
            out[:, j] = np.log(np.clip(out[:, j], _TINY, None))
    return out


@dataclass
class HierarchicalGenerator:
    """Fitted generator; build it with :meth:`fit`."""

    features: tuple[str, ...]
    log_features: frozenset[str]
    x_center: float
    mu: np.ndarray  # (2F,) mean of [intercepts, slopes]
    deviations: np.ndarray  # (k, 2F) pilot subjects minus mu
    resid_cov: np.ndarray  # (F, F) within-subject residual covariance (model scale)
    n_subjects: int
    variance_uncertainty: bool = True
    _chol: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        f = len(self.features)
        self._chol = np.linalg.cholesky(self.resid_cov + _TINY * np.eye(f))

    @property
    def n_features(self) -> int:
        return len(self.features)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        features: Sequence[str],
        group: str,
        x: str,
        *,
        log_features: Sequence[str] = (),
        x_center: float | None = None,
        dof_correction: bool = True,
        variance_uncertainty: bool = True,
        min_rows: int = 3,
    ) -> HierarchicalGenerator:
        """Fit the generator on a table of pilot subjects.

        Subjects with fewer than ``min_rows`` rows, or with a constant ``x`` (no slope can be
        estimated), are skipped with a warning. ``dof_correction`` rescales each subject's
        residuals by ``sqrt(n / (n - 2))`` to account for the two parameters fitted per subject;
        without it the residual variance is underestimated when subjects have few rows.
        """
        cols = tuple(features)
        log_set = frozenset(log_features)
        if not log_set <= set(cols):
            raise ValueError(f"log_features not in features: {sorted(log_set - set(cols))}")
        centre = float(df[x].mean()) if x_center is None else float(x_center)

        thetas: list[np.ndarray] = []
        residuals: list[np.ndarray] = []
        skipped: list[object] = []
        for g, d in df.groupby(group, sort=False):
            xv = d[x].to_numpy(dtype=float)
            if len(d) < min_rows or np.ptp(xv) == 0:
                skipped.append(g)
                continue
            y = _to_model_scale(d[list(cols)].to_numpy(dtype=float), cols, log_set)
            design = np.column_stack([np.ones_like(xv), xv - centre])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)  # (2, F)
            res = y - design @ coef
            if dof_correction:
                res = res * np.sqrt(len(d) / (len(d) - 2))
            thetas.append(np.concatenate([coef[0], coef[1]]))
            residuals.append(res)
        if skipped:
            warnings.warn(f"subjects skipped (too few rows or constant x): {skipped}", stacklevel=2)
        k = len(thetas)
        if k < 2:
            raise ValueError("need at least 2 usable subjects to estimate between-subject spread")
        if k < 4:
            warnings.warn(
                f"only {k} pilot subjects: the between-subject spread is very uncertain",
                stacklevel=2,
            )

        theta = np.asarray(thetas)
        mu = theta.mean(axis=0)
        res_all = np.vstack(residuals)
        sd = res_all.std(axis=0, ddof=1) + _TINY
        shrunk = LedoitWolf().fit(res_all / sd).covariance_
        scale = np.sqrt(np.diag(shrunk))
        corr = shrunk / np.outer(scale, scale)
        return cls(
            features=cols,
            log_features=log_set,
            x_center=centre,
            mu=mu,
            deviations=theta - mu,
            resid_cov=corr * np.outer(sd, sd),
            n_subjects=k,
            variance_uncertainty=variance_uncertainty,
        )

    # ----------------------------------------------------------------- sampling
    def sample_theta(
        self, rng: np.random.Generator, size: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intercepts ``a`` and slopes ``b`` of new subject(s): ``(F,)`` each, or ``(size, F)``."""
        k, f = self.n_subjects, self.n_features
        n = 1 if size is None else size
        s2 = (k - 1) / rng.chisquare(k - 1, size=n) if self.variance_uncertainty else np.ones(n)
        w = rng.standard_normal((n, k)) * np.sqrt(s2 / (k - 1))[:, None]
        theta = self.mu + w @ self.deviations
        a, b = theta[:, :f], theta[:, f:]
        return (a[0], b[0]) if size is None else (a, b)

    def simulate_model_scale(
        self, a: np.ndarray, b: np.ndarray, x: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Trials of one subject on the model (log) scale: ``(len(x), F)``."""
        xc = np.asarray(x, dtype=float) - self.x_center
        noise = rng.standard_normal((len(xc), self.n_features)) @ self._chol.T
        return a[None, :] + xc[:, None] * b[None, :] + noise

    def to_original_scale(self, values: np.ndarray) -> np.ndarray:
        """Undo the log transform of the log features."""
        out = np.array(values, dtype=float, copy=True)
        for j, name in enumerate(self.features):
            if name in self.log_features:
                out[:, j] = np.exp(out[:, j])
        return out

    def simulate(
        self, a: np.ndarray, b: np.ndarray, x: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Trials of one subject on the original feature scale: ``(len(x), F)``."""
        return self.to_original_scale(self.simulate_model_scale(a, b, x, rng))

    def simulate_cohort(
        self,
        x_per_subject: Sequence[np.ndarray],
        labels_per_subject: Sequence[np.ndarray],
        rng: np.random.Generator,
        *,
        prefix: str = "SIM",
    ) -> pd.DataFrame:
        """Draw one new subject per entry; returns a table (``subject``, ``label``, ``x``, ...)."""
        frames = []
        for i, (x, labels) in enumerate(zip(x_per_subject, labels_per_subject, strict=True), 1):
            a, b = self.sample_theta(rng)
            frame = pd.DataFrame(self.simulate(a, b, x, rng), columns=list(self.features))
            frame.insert(0, "x", np.asarray(x, dtype=float))
            frame.insert(0, "label", np.asarray(labels))
            frame.insert(0, "subject", f"{prefix}{i:02d}")
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    # -------------------------------------------------------------- inspection
    def parameter_table(self) -> pd.DataFrame:
        """Fitted population parameters, one row per feature."""
        f = self.n_features
        spread = self.deviations.std(axis=0, ddof=1)
        return pd.DataFrame(
            {
                "feature": list(self.features),
                "scale": ["log" if c in self.log_features else "linear" for c in self.features],
                "intercept_mean": self.mu[:f],
                "intercept_sd_between": spread[:f],
                "slope_mean": self.mu[f:],
                "slope_sd_between": spread[f:],
                "residual_sd": np.sqrt(np.diag(self.resid_cov)),
            }
        )

    def spread_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Chi-square interval for the between-subject SD, as multipliers of the estimate."""
        k = self.n_subjects
        lo, hi = chi2.ppf([(1 - level) / 2, (1 + level) / 2], k - 1)
        return float(np.sqrt((k - 1) / hi)), float(np.sqrt((k - 1) / lo))

    def scale_effect(self, factor: float) -> HierarchicalGenerator:
        """Copy whose slopes (mean and spread) are multiplied by ``factor``.

        ``factor = 0`` removes every dependence on ``x``: a null generator for checking false
        positive rates. ``factor < 1`` explores a weaker effect than the pilot suggests.
        """
        f = self.n_features
        mu = self.mu.copy()
        dev = self.deviations.copy()
        mu[f:] *= factor
        dev[:, f:] *= factor
        return HierarchicalGenerator(
            features=self.features,
            log_features=self.log_features,
            x_center=self.x_center,
            mu=mu,
            deviations=dev,
            resid_cov=self.resid_cov,
            n_subjects=self.n_subjects,
            variance_uncertainty=self.variance_uncertainty,
        )


def calibration_check(
    df: pd.DataFrame,
    features: Sequence[str],
    group: str,
    x: str,
    *,
    log_features: Sequence[str] = (),
    n_sim: int = 300,
    seed: int = 0,
    x_center: float | None = None,
) -> pd.DataFrame:
    """Leave-one-subject-out calibration of the generator.

    For every pilot subject ``h``: fit the generator on the others, simulate ``n_sim`` new subjects
    that perform the same ``x`` values as ``h``, and record the percentile of ``h``'s real mean of
    each feature among the simulated means. A calibrated generator spreads these percentiles
    roughly uniformly, so about 90% should land between 5 and 95. Many extreme percentiles mean the
    generator underestimates the difference between people.
    """
    cols = list(features)
    log_set = frozenset(log_features)
    centre = float(df[x].mean()) if x_center is None else float(x_center)
    rng = np.random.default_rng(seed)
    rows = []
    for held in pd.unique(df[group]):
        train = df[df[group] != held]
        test = df[df[group] == held]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gen = HierarchicalGenerator.fit(
                    train, cols, group, x, log_features=log_features, x_center=centre
                )
        except ValueError:
            continue
        xv = test[x].to_numpy(dtype=float)
        real = _to_model_scale(test[cols].to_numpy(dtype=float), cols, log_set).mean(axis=0)
        a, b = gen.sample_theta(rng, size=n_sim)  # (n_sim, F)
        noise = rng.standard_normal((n_sim, len(xv), len(cols))) @ gen._chol.T
        sims = (a[:, None, :] + (xv - centre)[None, :, None] * b[:, None, :] + noise).mean(axis=1)
        pct = (sims < real[None, :]).mean(axis=0) * 100
        rows.append(
            {"held_out": held, "k_train": gen.n_subjects, **dict(zip(cols, pct, strict=True))}
        )
    return pd.DataFrame(rows)


def fraction_calibrated(calibration: pd.DataFrame, lo: float = 5.0, hi: float = 95.0) -> float:
    """Share of percentiles inside ``[lo, hi]``: about 0.90 for a calibrated generator."""
    values = calibration.drop(columns=["held_out", "k_train"]).to_numpy(dtype=float).ravel()
    if values.size == 0:
        return float("nan")
    return float(np.mean((values >= lo) & (values <= hi)))
