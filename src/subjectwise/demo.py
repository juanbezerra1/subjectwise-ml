"""End-to-end demo on a synthetic cohort (``subjectwise demo``).

Everything here is simulated: no real participant data is involved. The demo walks through the
questions the package answers and writes a report, tables, figures and a summary file:

1. is the table sound (audit)?
2. how optimistic is a random split compared with leave-one-subject-out (LOSO)?
3. is the honest accuracy above chance (within-subject permutation test)?
4. which subjects drive it (leave-one-subject-out sensitivity)?
5. what does the correlation with a covariate look like when subjects are respected?
6. can new, plausible subjects be simulated from a small pilot (calibration)?
7. how many subjects would be needed (simulation-based power)?
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__, viz
from .checks import audit
from .correlation import correlation_report
from .cv import compare_schemes, evaluate, leakage_gap
from .generator import HierarchicalGenerator, calibration_check, fraction_calibrated
from .models import MODEL_LABELS
from .permutation import permutation_test
from .power import level_x_stats, power_curve
from .sensitivity import leave_group_out
from .toy import toy_cohort, toy_features

LABEL, GROUP, COVARIATE = "level", "subject", "x"
#: Share of the pilot's slopes kept in the power analysis: pilots tend to overstate effects.
EFFECT_FACTOR = 0.15


@dataclass(frozen=True)
class DemoSize:
    """How much work the demo does."""

    n_subjects: int
    trials_per_level: int
    n_perm: int
    n_sim: int
    power_subjects: tuple[int, ...]
    power_trials: tuple[int, ...]
    power_reps: int
    null_reps: int


FULL = DemoSize(8, 5, 500, 300, (4, 6, 8, 10, 12), (2, 5), 100, 200)
QUICK = DemoSize(6, 4, 60, 60, (4, 8), (2,), 12, 40)


def _md(df: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    """Minimal GitHub-flavoured markdown table (floats use ``formats[col]`` or ``.3f``)."""
    formats = formats or {}
    cols = list(df.columns)

    def cell(col: str, value: object) -> str:
        if isinstance(value, float | np.floating):
            return "" if np.isnan(value) else format(float(value), formats.get(col, ".3f"))
        return str(value)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(c, v) for c, v in zip(cols, row, strict=True)) + " |")
    return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def _pct(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        out[c] = out[c] * 100
    return out


def run_demo(
    out: str | Path,
    *,
    quick: bool = False,
    seed: int = 0,
    n_jobs: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the demo, write its files under ``out`` and return the summary dictionary.

    ``quick=True`` shrinks every simulation (about ten seconds); the numbers are then rough.
    """
    log = progress or (lambda message: None)
    size = QUICK if quick else FULL
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    df = toy_cohort(size.n_subjects, trials_per_level=size.trials_per_level, seed=seed)
    features, log_features = toy_features()
    df.to_csv(out / "toy_cohort.csv", index=False)
    chance = 1.0 / df[LABEL].nunique()

    log("1/7 auditing the table")
    report = audit(df, features, LABEL, GROUP)

    log("2/7 comparing validation schemes")
    models = ("lda", "svm_rbf")
    schemes = compare_schemes(
        df, features, LABEL, GROUP, models=models, log_features=log_features, random_state=seed
    )
    gap = leakage_gap(schemes)
    honest = evaluate(
        df, features, LABEL, GROUP, model="lda", scheme="loso",
        normalize="within_group", log_features=log_features, random_state=seed,
    )  # fmt: skip

    log(f"3/7 permutation test ({size.n_perm} shuffles)")
    perm = permutation_test(
        df, features, LABEL, GROUP, model="lda", normalize="within_group",
        log_features=log_features, n_perm=size.n_perm, seed=seed, n_jobs=n_jobs,
    )  # fmt: skip

    log("4/7 sensitivity to individual subjects")
    sensitivity = leave_group_out(
        df, features, LABEL, GROUP, model="lda", normalize="within_group",
        log_features=log_features, random_state=seed,
    )  # fmt: skip

    log("5/7 correlation with the covariate")
    corr_frame = df.assign(log_amp_1=np.log(df["amp_1"]))
    corr = correlation_report(corr_frame, COVARIATE, "log_amp_1", GROUP)

    log("6/7 fitting the simulated-subject generator and checking its calibration")
    generator = HierarchicalGenerator.fit(df, features, GROUP, COVARIATE, log_features=log_features)
    calibration = calibration_check(
        df, features, GROUP, COVARIATE, log_features=log_features, n_sim=size.n_sim, seed=seed
    )
    frac_ok = fraction_calibrated(calibration)

    log("7/7 power analysis by simulation")
    x_stats = level_x_stats(df, LABEL, COVARIATE)
    power = power_curve(
        generator.scale_effect(EFFECT_FACTOR), x_stats,
        n_subjects_list=size.power_subjects, trials_per_level_list=size.power_trials,
        n_reps=size.power_reps, null_reps=size.null_reps, normalize="within_group", seed=seed,
        progress=lambda message: log(f"    power: {message}"),
    )  # fmt: skip
    false_positives = power.false_positive_rates()

    # ---------------------------------------------------------------- tables and figures
    tables = out / "tables"
    schemes.to_csv(tables / "schemes.csv", index=False)
    gap.to_csv(tables / "leakage_gap.csv", index=False)
    honest.per_group().to_csv(tables / "per_subject.csv", index=False)
    honest.confusion().to_csv(tables / "confusion.csv")
    sensitivity.to_csv(tables / "sensitivity.csv", index=False)
    corr.per_group.to_csv(tables / "correlation_per_subject.csv", index=False)
    generator.parameter_table().to_csv(tables / "generator_parameters.csv", index=False)
    calibration.to_csv(tables / "calibration.csv", index=False)
    power.summary.to_csv(tables / "power.csv", index=False)
    false_positives.to_csv(tables / "false_positive_check.csv", index=False)

    figures = out / "figures"
    viz.save_figure(viz.plot_scheme_comparison(schemes, chance=chance), figures / "schemes.png")
    viz.save_figure(viz.plot_subject_accuracy(honest, perm=perm), figures / "subjects.png")
    viz.save_figure(viz.plot_permutation(perm), figures / "permutation.png")
    viz.save_figure(viz.plot_calibration(calibration), figures / "calibration.png")
    viz.save_figure(
        viz.plot_power(
            power.summary,
            title=f"Power if the true effect is {EFFECT_FACTOR:.0%} of the pilot's",
        ),
        figures / "power.png",
    )

    # ------------------------------------------------------------------------- summary
    lda_naive = float(gap.loc[gap["model"] == "lda", "kfold_accuracy"].iloc[0])
    lda_loso = float(gap.loc[gap["model"] == "lda", "loso_accuracy"].iloc[0])
    summary: dict[str, Any] = {
        "version": __version__,
        "seed": seed,
        "quick": quick,
        "cohort": {
            "n_subjects": int(df[GROUP].nunique()),
            "n_rows": len(df),
            "classes": sorted(df[LABEL].unique().tolist()),
            "chance": chance,
        },
        "audit": {"ok": report.ok, "errors": len(report.errors), "warnings": len(report.warnings)},
        "schemes": schemes.to_dict(orient="records"),
        "leakage_gap": gap.to_dict(orient="records"),
        "honest": honest.summary(),
        "permutation": perm.summary(),
        "sensitivity": {
            "min_accuracy": float(sensitivity["mean_accuracy"].iloc[1:].min()),
            "max_accuracy": float(sensitivity["mean_accuracy"].iloc[1:].max()),
        },
        "correlation": {
            "pooled": vars(corr.pooled),
            "within": vars(corr.within),
            "subject_level": vars(corr.subject_level),
        },
        "generator": {
            "n_pilot_subjects": generator.n_subjects,
            "fraction_calibrated": frac_ok,
            "sd_interval_95": generator.spread_interval(),
        },
        "power": power.summary.to_dict(orient="records"),
        "false_positive_check": false_positives.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    # --------------------------------------------------------------------------- report
    lines: list[str] = []
    add = lines.append
    n_sub, n_rows = df[GROUP].nunique(), len(df)
    levels = ", ".join(map(str, sorted(df[LABEL].unique())))
    add("# subjectwise demo report\n")
    add(
        f"Synthetic cohort: {n_sub} subjects, {n_rows} trials, {len(features)} features, "
        f"{df[LABEL].nunique()} classes (levels {levels}), "
        f"chance = {chance:.0%}. Seed {seed}, subjectwise {__version__}"
        f"{', quick mode (rough numbers)' if quick else ''}. All data are simulated.\n"
    )
    add("## 1. Data audit\n")
    add("```\n" + report.format() + "\n```\n")

    add("## 2. How you validate changes the answer\n")
    shown = _pct(schemes, ["mean_accuracy", "sd_accuracy", "balanced_accuracy"])
    shown["scenario"] = [
        viz.SCENARIO_LABELS.get((s, n), f"{s} / {n}")
        for s, n in zip(shown["scheme"], shown["normalize"], strict=True)
    ]
    shown["model"] = shown["model"].map(lambda m: MODEL_LABELS.get(m, m))
    add(
        _md(
            shown[
                ["model", "scenario", "mean_accuracy", "sd_accuracy", "balanced_accuracy"]
            ].rename(
                columns={
                    "mean_accuracy": "mean acc. (%)",
                    "sd_accuracy": "SD across subjects (pts)",
                    "balanced_accuracy": "balanced acc. (%)",
                }
            ),
            {"mean acc. (%)": ".1f", "SD across subjects (pts)": ".1f", "balanced acc. (%)": ".1f"},
        )  # fmt: skip
        + "\n"
    )
    add("![Validation schemes](figures/schemes.png)\n")
    gap_points = (lda_naive - lda_loso) * 100
    verdict = (
        f"the random split overstates accuracy by {gap_points:.1f} points"
        if gap_points >= 0
        else f"the random split happens to understate accuracy by {-gap_points:.1f} points here"
    )
    add(
        f"With LDA, a random split reports {lda_naive:.1%} and holding out whole subjects "
        f"{lda_loso:.1%}: {verdict}. Adding within-subject z-scoring (labels are never used) to "
        f"the honest scheme gives {honest.mean_accuracy:.1%}, with a standard deviation of "
        f"{honest.sd_accuracy * 100:.1f} points across subjects.\n"
    )

    add("## 3. Is it above chance?\n")
    add(
        f"Labels were shuffled {perm.n_perm} times *inside each subject* and the whole LOSO "
        f"pipeline was repeated. Observed mean per-subject accuracy: {perm.observed:.1%}. "
        f"Permutation null: mean {perm.null_mean:.1%}, 95th percentile {perm.null_p95:.1%}. "
        f"p = {perm.p_value:.4f} (the smallest attainable value is {1 / (perm.n_perm + 1):.4f}).\n"
    )
    add("![Permutation null](figures/permutation.png)\n")
    add("![Per-subject accuracy](figures/subjects.png)\n")

    add("## 4. Which subjects drive the result?\n")
    add(
        _md(
            _pct(sensitivity, ["mean_accuracy", "sd_accuracy", "balanced_accuracy"])[
                ["scenario", "n_groups", "mean_accuracy", "balanced_accuracy", "delta_points"]
            ].rename(
                columns={
                    "mean_accuracy": "mean acc. (%)",
                    "balanced_accuracy": "balanced acc. (%)",
                    "delta_points": "change (pts)",
                }
            ),
            {"mean acc. (%)": ".1f", "balanced acc. (%)": ".1f", "change (pts)": "+.1f"},
        )  # fmt: skip
        + "\n"
    )
    add(
        "A diagnostic, not a licence to drop subjects: exclusion rules must be fixed before "
        "looking at accuracy.\n"
    )

    add("## 5. Correlation with the covariate\n")
    add(
        _md(
            pd.DataFrame(
                [
                    {
                        "view": "pooled (all rows)",
                        "r": corr.pooled.r,
                        "p": corr.pooled.p,
                        "method": corr.pooled.method,
                        "n": corr.pooled.n,
                    },
                    {
                        "view": "within subject (centred)",
                        "r": corr.within.r,
                        "p": corr.within.p,
                        "method": corr.within.method,
                        "n": corr.within.n,
                    },
                    {
                        "view": "across subjects (Fisher-z)",
                        "r": corr.subject_level.r,
                        "p": corr.subject_level.p,
                        "method": corr.subject_level.method,
                        "n": corr.subject_level.n,
                    },
                ]
            ),
            {"p": ".2g"},
        )  # fmt: skip
        + "\n"
    )
    add(
        "`n` counts rows, rows and subjects respectively. Only the last p-value respects the "
        "subject structure; the first two treat every trial as independent.\n"
    )

    add("## 6. Simulated subjects\n")
    add(
        f"The generator was fitted on {generator.n_subjects} subjects. Leave-one-subject-out "
        f"calibration: {frac_ok:.0%} of the real subjects' feature means fall inside the 5-95 "
        "percentile band of simulated ones (about 90% is expected). With few pilot subjects the "
        f"between-subject SD is known only within a factor of {generator.spread_interval()[0]:.2f} "
        f"to {generator.spread_interval()[1]:.2f} (95% interval).\n"
    )
    add(
        _md(
            generator.parameter_table(),
            {c: ".3f" for c in generator.parameter_table().columns},
        )
        + "\n"
    )
    add("![Calibration](figures/calibration.png)\n")

    add("## 7. How many subjects?\n")
    add(
        f"Studies were simulated from the generator with the pilot's slopes multiplied by "
        f"{EFFECT_FACTOR} (a pessimistic effect, because pilots tend to overstate) and analysed "
        f"with LOSO + within-subject z-scoring. A study counts as detected when its mean accuracy "
        f"is above the 95th percentile of the same statistic in {size.null_reps} studies simulated "
        f"*without* any effect (`null threshold`), so the false-positive rate is 5% by "
        f"construction. {size.power_reps} studies per design; the two classical tests are shown "
        "for comparison.\n"
    )
    add(
        _md(
            power.summary.rename(
                columns={
                    "trials_per_level": "trials/class",
                    "n_subjects": "subjects",
                    "acc_median": "median acc.",
                    "acc_p05": "acc. p5",
                    "acc_p95": "acc. p95",
                    "null_threshold": "null threshold",
                    "power_calibrated": "power (calibrated)",
                    "power_ttest": "power (t-test)",
                    "power_binom": "power (binomial)",
                }
            ).drop(columns="model"),
            {
                c: ".2f"
                for c in (
                    "median acc.",
                    "acc. p5",
                    "acc. p95",
                    "null threshold",
                    "power (calibrated)",
                    "power (t-test)",
                    "power (binomial)",
                )
            },
        )  # fmt: skip
        + "\n"
    )
    add("![Power](figures/power.png)\n")
    add(
        "False-positive check of the two classical tests: the same pipeline with the effect "
        "removed. Values above 0.05 mean the test is liberal for this pipeline, so the power "
        "it reports is optimistic.\n"
    )
    add(
        _md(
            false_positives.rename(
                columns={
                    "trials_per_level": "trials/class",
                    "n_subjects": "subjects",
                    "null_acc_median": "median acc.",
                    "fpr_binom": "false positives (binomial)",
                    "fpr_ttest": "false positives (t-test)",
                    "n_null": "null studies",
                }
            ).drop(columns="model"),
            {
                "median acc.": ".2f",
                "false positives (binomial)": ".3f",
                "false positives (t-test)": ".3f",
            },
        )  # fmt: skip
        + "\n"
    )

    add("## Caveats\n")
    add("- The toy cohort is not physiology: it only has the statistical shape of small studies.")
    add("- Within-subject z-scoring is transductive: each subject's own test rows contribute to")
    add("  its mean and SD. It never uses labels, but it assumes the subject's rows are available.")
    add("- The generator is Gaussian, linear in the covariate and can only reproduce what the")
    add("  pilot subjects show; with few subjects every power curve is itself uncertain.")
    add("- A sanity check on the analysis, not a substitute for a pre-registered sample size.")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"done: {out / 'report.md'}")
    return summary
