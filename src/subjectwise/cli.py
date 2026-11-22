"""Command-line interface: ``subjectwise demo | check | evaluate | power``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import pandas as pd
import typer

from . import __version__, viz
from .checks import audit
from .cv import DEFAULT_SCENARIOS, compare_schemes, leakage_gap
from .cv import evaluate as evaluate_cv
from .demo import run_demo
from .generator import HierarchicalGenerator, calibration_check, fraction_calibrated
from .models import MODEL_NAMES
from .permutation import permutation_test
from .power import level_x_stats, power_curve
from .sensitivity import leave_group_out
from .tables import infer_features, read_table, require_columns, split_names

app = typer.Typer(
    help="Honest validation for classifiers trained on a handful of subjects.",
    no_args_is_help=True,
    add_completion=False,
)

TableArg = Annotated[
    Path,
    typer.Argument(exists=True, dir_okay=False, help="CSV with one row per trial or window."),
]
LabelOpt = Annotated[str, typer.Option(help="Class label column.")]
GroupOpt = Annotated[str, typer.Option(help="Subject (group) column.")]
FeaturesOpt = Annotated[
    str | None,
    typer.Option(
        help="Comma-separated feature columns. Default: every numeric column that is not "
        "bookkeeping (subject, label, level, x, trial, ...)."
    ),
]
LogOpt = Annotated[
    str | None,
    typer.Option(
        "--log-features",
        help="Comma-separated features that are positive and multiplicative (RMS, power): "
        "they are log-transformed before the within-subject z-score.",
    ),
]
SeedOpt = Annotated[int, typer.Option(help="Random seed.")]
OutOpt = Annotated[
    Path | None, typer.Option("--out", "-o", help="Directory for CSV tables and figures.")
]


def _fail(message: str) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _show(df: pd.DataFrame, max_rows: int = 60) -> None:
    typer.echo(df.to_string(index=False, max_rows=max_rows, float_format=lambda v: f"{v:.4g}"))


def _version(value: bool) -> None:
    if value:
        typer.echo(f"subjectwise {__version__}")
        raise typer.Exit()


def _load(
    table: Path,
    label: str,
    group: str,
    features: str | None,
    log_features: str | None,
    extra: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Read the table and resolve feature columns; exit with a clear message on bad input."""
    try:
        df = read_table(table)
        require_columns(df, [label, group, *extra])
        cols = split_names(features) or infer_features(df, exclude=[label, group, *extra])
        require_columns(df, cols)
        logs = split_names(log_features)
        if not set(logs) <= set(cols):
            raise ValueError(
                f"--log-features not among the features: {sorted(set(logs) - set(cols))}"
            )
    except (OSError, ValueError) as exc:
        _fail(str(exc))
    if not cols:
        _fail("no feature columns found; pass --features")
    return df, cols, logs


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """subjectwise command-line interface."""


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path(
        "subjectwise_demo"
    ),
    seed: SeedOpt = 0,
    quick: Annotated[bool, typer.Option(help="Shrink every simulation (rough numbers).")] = False,
    jobs: Annotated[int, typer.Option(help="Parallel workers for the permutation test.")] = 1,
) -> None:
    """Run every analysis on a synthetic cohort; write report, tables and figures."""
    summary = run_demo(out, quick=quick, seed=seed, n_jobs=jobs, progress=typer.echo)
    honest, perm = summary["honest"], summary["permutation"]
    typer.echo(
        f"\nLOSO + within-subject z-score: {honest['mean_accuracy']:.1%} "
        f"(permutation p = {perm['p_value']:.4f}).  Report: {out / 'report.md'}"
    )


@app.command()
def check(
    table: TableArg,
    label: LabelOpt = "level",
    group: GroupOpt = "subject",
    features: FeaturesOpt = None,
    min_rows: Annotated[int, typer.Option(help="Rows below which a subject is flagged.")] = 3,
    strict: Annotated[bool, typer.Option(help="Also exit with status 1 on warnings.")] = False,
) -> None:
    """Audit a table before modelling: missing classes, tiny or duplicated subjects, ..."""
    df, cols, _ = _load(table, label, group, features, None)
    report = audit(df, cols, label, group, min_rows_per_group=min_rows)
    typer.echo(report.format())
    if report.errors or (strict and report.warnings):
        raise typer.Exit(1)


@app.command()
def evaluate(
    table: TableArg,
    label: LabelOpt = "level",
    group: GroupOpt = "subject",
    features: FeaturesOpt = None,
    log_features: LogOpt = None,
    models: Annotated[
        str, typer.Option(help=f"Comma-separated, from {', '.join(MODEL_NAMES)}.")
    ] = "lda,svm_rbf",
    normalize: Annotated[
        str, typer.Option(help="Normalisation of the permutation test: none | within_group.")
    ] = "within_group",
    permutations: Annotated[int, typer.Option(help="Permutations (0 skips the test).")] = 200,
    seed: SeedOpt = 0,
    jobs: Annotated[int, typer.Option(help="Parallel workers for the permutation test.")] = 1,
    out: OutOpt = None,
) -> None:
    """Random split vs leave-one-subject-out, permutation test and per-subject sensitivity."""
    df, cols, logs = _load(table, label, group, features, log_features)
    names = split_names(models)
    unknown = [m for m in names if m not in MODEL_NAMES]
    if unknown or not names:
        _fail(f"unknown model(s) {unknown}; choose from {list(MODEL_NAMES)}")
    if normalize not in ("none", "within_group"):
        _fail("--normalize must be 'none' or 'within_group'")

    report = audit(df, cols, label, group)
    typer.echo(report.format())
    if report.errors:
        _fail("the table has errors (see above); fix them or run `subjectwise check`")

    try:
        schemes = compare_schemes(
            df, cols, label, group, models=names, scenarios=DEFAULT_SCENARIOS,
            log_features=logs, random_state=seed,
        )  # fmt: skip
        typer.echo("\nValidation schemes (mean of per-subject accuracies):")
        _show(
            schemes[
                [
                    "model",
                    "scheme",
                    "normalize",
                    "mean_accuracy",
                    "sd_accuracy",
                    "balanced_accuracy",
                ]
            ]
        )
        typer.echo("\nOptimism of the random split (points):")
        _show(leakage_gap(schemes))

        honest = evaluate_cv(
            df, cols, label, group, model=names[0], scheme="loso", normalize=normalize,  # type: ignore[arg-type]
            log_features=logs, random_state=seed,
        )  # fmt: skip
        typer.echo(f"\nPer-subject accuracy ({names[0]}, LOSO, normalize={normalize}):")
        _show(honest.per_group())
        perm = None
        if permutations > 0:
            perm = permutation_test(
                df, cols, label, group, model=names[0], normalize=normalize, log_features=logs,
                n_perm=permutations, seed=seed, n_jobs=jobs,
            )  # fmt: skip
            s = perm.summary()
            typer.echo(
                f"\nPermutation test ({permutations} shuffles inside each subject): observed "
                f"{s['observed']:.1%}, null mean {s['null_mean']:.1%}, null 95th pct "
                f"{s['null_p95']:.1%}, p = {s['p_value']:.4f}"
            )
        sensitivity = None
        if df[group].nunique() >= 4:
            sensitivity = leave_group_out(
                df, cols, label, group, model=names[0], normalize=normalize, log_features=logs,
                random_state=seed,
            )  # fmt: skip
            typer.echo("\nSensitivity to individual subjects:")
            _show(sensitivity)
    except ValueError as exc:
        _fail(str(exc))

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        schemes.to_csv(out / "schemes.csv", index=False)
        honest.per_group().to_csv(out / "per_subject.csv", index=False)
        if sensitivity is not None:
            sensitivity.to_csv(out / "sensitivity.csv", index=False)
        chance = 1.0 / df[label].nunique()
        viz.save_figure(viz.plot_scheme_comparison(schemes, chance=chance), out / "schemes.png")
        viz.save_figure(
            viz.plot_subject_accuracy(honest, chance=chance, perm=perm), out / "subjects.png"
        )
        if perm is not None:
            viz.save_figure(viz.plot_permutation(perm), out / "permutation.png")
        typer.echo(f"\nFiles written to {out}")


@app.command()
def power(
    table: TableArg,
    label: LabelOpt = "level",
    group: GroupOpt = "subject",
    x: Annotated[str, typer.Option(help="Numeric covariate column (e.g. produced force).")] = "x",
    features: FeaturesOpt = None,
    log_features: LogOpt = None,
    subjects: Annotated[
        str, typer.Option(help="Comma-separated numbers of subjects.")
    ] = "4,6,8,10,12",
    trials: Annotated[str, typer.Option(help="Comma-separated trials per class.")] = "2,5",
    reps: Annotated[int, typer.Option(help="Simulated studies per design.")] = 100,
    effect: Annotated[
        float, typer.Option(help="Multiply the pilot's slopes by this factor (1 = as observed).")
    ] = 1.0,
    normalize: Annotated[str, typer.Option(help="none | within_group.")] = "within_group",
    model: Annotated[str, typer.Option(help=f"One of {', '.join(MODEL_NAMES)}.")] = "lda",
    seed: SeedOpt = 0,
    out: OutOpt = None,
) -> None:
    """Simulate studies from a pilot table and estimate power versus number of subjects."""
    df, cols, logs = _load(table, label, group, features, log_features, extra=(x,))
    if model not in MODEL_NAMES:
        _fail(f"unknown model {model!r}; choose from {list(MODEL_NAMES)}")
    if normalize not in ("none", "within_group"):
        _fail("--normalize must be 'none' or 'within_group'")
    try:
        n_list = [int(v) for v in split_names(subjects)]
        t_list = [int(v) for v in split_names(trials)]
        if not n_list or not t_list or min(n_list) < 2 or min(t_list) < 1:
            raise ValueError("--subjects needs values >= 2 and --trials values >= 1")
        generator = HierarchicalGenerator.fit(df, cols, group, x, log_features=logs)
        typer.echo(f"Generator fitted on {generator.n_subjects} pilot subjects.")
        low, high = generator.spread_interval()
        typer.echo(f"Between-subject SD is uncertain by a factor of {low:.2f} to {high:.2f} (95%).")
        calibration = calibration_check(df, cols, group, x, log_features=logs, seed=seed)
        typer.echo(
            f"Calibration: {fraction_calibrated(calibration):.0%} of held-out subjects' feature "
            "means fall inside the 5-95 percentile band of simulated subjects (about 90% expected)."
        )
        result = power_curve(
            generator.scale_effect(effect), level_x_stats(df, label, x),
            n_subjects_list=n_list, trials_per_level_list=t_list, n_reps=reps, models=(model,),
            normalize=normalize, seed=seed,
        )  # fmt: skip
    except ValueError as exc:
        _fail(str(exc))
    typer.echo(
        f"\nPower by design (effect x {effect:g}, {reps} simulated studies per row). "
        "`power_calibrated` counts studies whose accuracy beats the 95th percentile of simulated "
        "null studies (`null_threshold`):"
    )
    _show(result.summary)
    typer.echo("\nFalse positives of the two classical tests when the effect is removed:")
    _show(result.false_positive_rates())
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        result.summary.to_csv(out / "power.csv", index=False)
        result.false_positive_rates().to_csv(out / "false_positive_check.csv", index=False)
        calibration.to_csv(out / "calibration.csv", index=False)
        viz.save_figure(viz.plot_power(result.summary), out / "power.png")
        viz.save_figure(viz.plot_calibration(calibration), out / "calibration.png")
        typer.echo(f"\nFiles written to {out}")
