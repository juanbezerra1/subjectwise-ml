"""Simulation-based power analysis for subject-wise classification.

"How many subjects do I need before leave-one-subject-out accuracy is reliably above chance?" has no
closed-form answer, because the answer depends on how different people are. This module answers it
by simulation: draw complete synthetic studies from a
:class:`~subjectwise.generator.HierarchicalGenerator` (fitted on pilot data), run the real LOSO
pipeline on each, and count how often the result is significant.

Three decision rules are reported for every design (number of subjects x trials per class):

``power_calibrated`` (recommended)
    A study counts as a success when its mean per-subject accuracy exceeds the ``1 - alpha``
    quantile of the same statistic in studies simulated with the effect removed
    (``scale_effect(0)``) for the *same design and pipeline*. Its false-positive rate is therefore
    at most ``alpha`` by construction (up to the Monte-Carlo error of the simulated null), whatever
    the model and the normalisation. ``null_threshold`` is the accuracy a study must beat.
``power_ttest``
    One-sided one-sample t-test of the per-subject accuracies against chance (``1/K``). Subjects are
    independent, so it matches the question "will it work on a new person?", but accuracies from a
    handful of subjects are discrete and bounded, and LOSO folds share training data.
``power_binom``
    Binomial test of the pooled number of correct predictions against chance. It treats every row as
    an independent draw, which they are not.

The two tests are approximations. :meth:`PowerResult.false_positive_rates` measures how liberal or
conservative each of them is for the pipeline at hand, using the simulated null studies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .cv import _balanced, group_accuracies, predict_loso, prepare
from .generator import HierarchicalGenerator

_KEYS = ["model", "trials_per_level", "n_subjects"]
_TOL = 1e-12


def level_x_stats(df: pd.DataFrame, label: str, x: str) -> pd.DataFrame:
    """Mean and SD of the covariate ``x`` for every class (used to simulate realistic ``x``)."""
    out = df.groupby(label)[x].agg(["mean", "std"]).rename(columns={"std": "sd"})
    out["sd"] = out["sd"].fillna(0.0)
    return out


def _one_sided_t(values: np.ndarray, chance: float) -> float:
    """P-value of ``mean(values) > chance``; the degenerate case of identical values is explicit.

    When every subject has exactly the same accuracy the t statistic is infinite (above chance) or
    undefined (equal to chance); scipy would warn about catastrophic cancellation there.
    """
    if len(values) < 2:
        return float("nan")
    if np.ptp(values) == 0:
        return 0.0 if values[0] > chance else 1.0
    return float(stats.ttest_1samp(values, chance, alternative="greater").pvalue)


@dataclass
class PowerResult:
    """Simulated studies and their summary per design.

    ``runs`` has one row per simulated study with the effect; ``null_runs`` the same without the
    effect (empty when ``calibrate=False``); ``summary`` one row per design.
    """

    runs: pd.DataFrame
    summary: pd.DataFrame
    null_runs: pd.DataFrame = field(default_factory=pd.DataFrame)
    alpha: float = 0.05

    def false_positive_rates(self) -> pd.DataFrame:
        """Share of *null* studies that each test declares significant, per design.

        Values near ``alpha`` mean the test is well calibrated for this pipeline; clearly larger
        values mean it is liberal and its power is optimistic.
        """
        if self.null_runs.empty:
            raise ValueError("no null studies were simulated (calibrate=False)")
        out = (
            self.null_runs.groupby(_KEYS)
            .agg(
                null_acc_median=("mean_accuracy", "median"),
                fpr_binom=("p_binom", lambda v: float(np.mean(v < self.alpha))),
                fpr_ttest=("p_ttest", lambda v: float(np.mean(v < self.alpha))),
                n_null=("rep", "size"),
            )
            .reset_index()
        )
        return out


def _simulate_designs(
    generator: HierarchicalGenerator,
    x_stats: pd.DataFrame,
    *,
    n_subjects_list: Sequence[int],
    trials_per_level_list: Sequence[int],
    n_reps: int,
    models: Sequence[str],
    normalize: str,
    rng: np.random.Generator,
    progress: Callable[[str], None] | None,
    tag: str,
) -> pd.DataFrame:
    classes = list(x_stats.index)
    chance = 1.0 / len(classes)
    features = list(generator.features)
    log_features = sorted(generator.log_features)
    rows: list[dict[str, object]] = []
    for trials in trials_per_level_list:
        labels = np.repeat(np.asarray(classes, dtype=object), trials)
        mean = x_stats.loc[labels, "mean"].to_numpy(dtype=float)
        sd = x_stats.loc[labels, "sd"].to_numpy(dtype=float)
        for n in n_subjects_list:
            for rep in range(n_reps):
                x_per_subject = [rng.normal(mean, sd) for _ in range(n)]
                table = generator.simulate_cohort(x_per_subject, [labels] * n, rng)
                xs, ys, gs = prepare(
                    table, features, "label", "subject",
                    normalize=normalize, log_features=log_features,
                )  # fmt: skip
                for name in models:
                    pred = predict_loso(xs, ys, gs, model=name)
                    per_subject = group_accuracies(ys, pred, gs)
                    hits = int(np.sum(pred == ys))
                    rows.append(
                        {
                            "model": name,
                            "trials_per_level": trials,
                            "n_subjects": n,
                            "rep": rep,
                            "mean_accuracy": float(per_subject.mean()),
                            "balanced_accuracy": _balanced(ys, pred),
                            "p_binom": float(
                                stats.binomtest(hits, len(ys), chance, alternative="greater").pvalue
                            ),
                            "p_ttest": _one_sided_t(per_subject, chance),
                        }
                    )
            if progress is not None:
                progress(f"{tag}: {trials} trials/class, N={n} done")
    return pd.DataFrame(rows)


def power_curve(
    generator: HierarchicalGenerator,
    x_stats: pd.DataFrame,
    *,
    n_subjects_list: Sequence[int] = (4, 6, 8, 10, 12),
    trials_per_level_list: Sequence[int] = (2, 5),
    n_reps: int = 100,
    null_reps: int | None = None,
    calibrate: bool = True,
    models: Sequence[str] = ("lda",),
    normalize: str = "none",
    alpha: float = 0.05,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> PowerResult:
    """Simulate ``n_reps`` studies for every ``(N subjects, trials per class)`` design.

    ``x_stats`` has one row per class with columns ``mean`` and ``sd`` (see :func:`level_x_stats`);
    its index defines the classes. Each class is repeated ``trials_per_level`` times per subject, so
    the design is balanced and chance is ``1 / n_classes``.

    With ``calibrate=True`` (default) ``null_reps`` studies per design (default: at least 200) are
    also simulated with the effect removed, which gives ``null_threshold`` and ``power_calibrated``
    in the summary and enables :meth:`PowerResult.false_positive_rates`.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if n_reps < 1:
        raise ValueError("n_reps must be at least 1")
    rng = np.random.default_rng(seed)
    common = {
        "n_subjects_list": n_subjects_list,
        "trials_per_level_list": trials_per_level_list,
        "models": models,
        "normalize": normalize,
        "rng": rng,
        "progress": progress,
    }
    runs = _simulate_designs(generator, x_stats, n_reps=n_reps, tag="effect", **common)  # type: ignore[arg-type]
    null_runs = pd.DataFrame()
    if calibrate:
        n_null = max(n_reps, 200) if null_reps is None else null_reps
        null_runs = _simulate_designs(
            generator.scale_effect(0.0),
            x_stats,
            n_reps=n_null,
            tag="null",
            **common,  # type: ignore[arg-type]
        )

    summary = (
        runs.groupby(_KEYS)
        .agg(
            acc_median=("mean_accuracy", "median"),
            acc_p05=("mean_accuracy", lambda v: float(np.percentile(v, 5))),
            acc_p95=("mean_accuracy", lambda v: float(np.percentile(v, 95))),
            power_binom=("p_binom", lambda v: float(np.mean(v < alpha))),
            power_ttest=("p_ttest", lambda v: float(np.mean(v < alpha))),
        )
        .reset_index()
    )
    if not null_runs.empty:
        threshold = (
            null_runs.groupby(_KEYS)["mean_accuracy"]
            .apply(lambda v: float(np.quantile(v, 1.0 - alpha, method="higher")))
            .rename("null_threshold")
            .reset_index()
        )
        merged = runs.merge(threshold, on=_KEYS)
        merged["above"] = merged["mean_accuracy"] > merged["null_threshold"] + _TOL
        calibrated = merged.groupby(_KEYS)["above"].mean().rename("power_calibrated").reset_index()
        summary = summary.merge(threshold, on=_KEYS).merge(calibrated, on=_KEYS)
        summary = summary[
            [*_KEYS, "acc_median", "acc_p05", "acc_p95", "null_threshold",
             "power_calibrated", "power_ttest", "power_binom"]
        ]  # fmt: skip
    return PowerResult(runs=runs, summary=summary, null_runs=null_runs, alpha=alpha)
