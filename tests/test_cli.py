import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import subjectwise as sw
from subjectwise.cli import app

runner = CliRunner()


def test_version_and_help():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and sw.__version__ in result.output
    help_text = runner.invoke(app, ["--help"]).output
    for command in ("demo", "check", "evaluate", "power"):
        assert command in help_text


def test_quick_demo_writes_report_tables_figures_and_summary(tmp_path):
    out = tmp_path / "demo"
    result = runner.invoke(app, ["demo", "--out", str(out), "--quick"])
    assert result.exit_code == 0, result.output
    report = (out / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## 1. Data audit",
        "## 3. Is it above chance?",
        "## 7. How many subjects?",
        "## 8. Tuning inside the training subjects",
    ):
        assert heading in report
    for name in ("schemes", "subjects", "permutation", "calibration", "power"):
        assert (out / "figures" / f"{name}.png").stat().st_size > 1000
    for name in (
        "schemes",
        "leakage_gap",
        "per_subject",
        "sensitivity",
        "power",
        "false_positive_check",
        "tuning",
    ):
        assert (out / "tables" / f"{name}.csv").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["cohort"]["chance"] == 0.25 and summary["quick"] is True
    assert 0.0 < summary["permutation"]["p_value"] <= 1.0
    tuning = summary["tuning"]
    assert tuning["n_candidates"] == 12 and tuning["model"] == "svm_rbf"
    assert tuning["optimism"] == pytest.approx(tuning["inner_accuracy"] - tuning["outer_accuracy"])
    assert "tuned, inner: what the choice was made on" in report
    cohort = pd.read_csv(out / "toy_cohort.csv")
    assert {"subject", "level", "x", "amp_1", "spec_1"} <= set(cohort.columns)


def test_demo_is_reproducible_for_a_given_seed(tmp_path):
    for name in ("a", "b"):
        assert (
            runner.invoke(
                app, ["demo", "--out", str(tmp_path / name), "--quick", "--seed", "4"]
            ).exit_code
            == 0
        )
    a = json.loads((tmp_path / "a" / "summary.json").read_text())
    b = json.loads((tmp_path / "b" / "summary.json").read_text())
    assert a == b


def test_check_passes_on_a_clean_table_and_fails_on_errors(toy_csv, tmp_path):
    ok = runner.invoke(app, ["check", toy_csv])
    assert ok.exit_code == 0 and "no issues found" in ok.output

    df = pd.read_csv(toy_csv)
    one = tmp_path / "one.csv"
    df[df["subject"] == "S01"].to_csv(one, index=False)
    bad = runner.invoke(app, ["check", str(one)])
    assert bad.exit_code == 1 and "too_few_groups" in bad.output


def test_check_strict_turns_warnings_into_a_failing_exit_code(toy_csv, tmp_path):
    df = pd.read_csv(toy_csv)
    twin = df[df["subject"] == "S02"].assign(subject="S02_copy")
    path = tmp_path / "dup.csv"
    pd.concat([df, twin]).to_csv(path, index=False)
    assert runner.invoke(app, ["check", str(path)]).exit_code == 0
    strict = runner.invoke(app, ["check", str(path), "--strict"])
    assert strict.exit_code == 1 and "duplicate_groups" in strict.output


def test_evaluate_prints_every_section_and_writes_files(toy_csv, tmp_path):
    out = tmp_path / "eval"
    result = runner.invoke(
        app,
        ["evaluate", toy_csv, "--log-features", "amp_1,amp_2,amp_3,amp_4", "--models", "lda",
         "--permutations", "10", "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    for text in (
        "Validation schemes",
        "Optimism of the random split",
        "Per-subject accuracy",
        "Permutation test",
        "Sensitivity to individual subjects",
    ):
        assert text in result.output
    assert (out / "schemes.csv").exists() and (out / "subjects.png").exists()
    assert (out / "permutation.png").exists() and (out / "sensitivity.csv").exists()


def test_evaluate_tune_shows_the_inner_score_next_to_the_outer_one(toy_csv, tmp_path):
    out = tmp_path / "tuned"
    result = runner.invoke(
        app,
        ["evaluate", toy_csv, "--log-features", "amp_1,amp_2,amp_3,amp_4", "--models", "svm_rbf",
         "--tune", "--permutations", "2", "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    for text in (
        "nested LOSO",
        "Nested tuning of svm_rbf",
        "fixed settings:",
        "tuned, outer (report it):",
        "tuned, inner (the choice was made on it):",
        "optimism of the inner score:",
        "tuning repeated in every shuffle",
        "Sensitivity to individual subjects",
    ):
        assert text in result.output
    tuning = pd.read_csv(out / "tuning.csv")
    assert len(tuning) == 6 and {"held_out", "param_C", "param_gamma", "inner_score"} <= set(
        tuning.columns
    )
    plain = runner.invoke(
        app,
        [
            "evaluate",
            toy_csv,
            "--models",
            "svm_rbf",
            "--permutations",
            "0",
            "--out",
            str(tmp_path / "p"),
        ],
    )
    assert plain.exit_code == 0 and "Nested tuning" not in plain.output
    assert not (tmp_path / "p" / "tuning.csv").exists()


def test_evaluate_tune_with_a_grouped_inner_loop(toy_csv):
    result = runner.invoke(
        app,
        ["evaluate", toy_csv, "--models", "svm_linear", "--tune", "--inner", "group_kfold",
         "--inner-splits", "2", "--permutations", "0"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "inner group_kfold" in result.output


def test_evaluate_tune_refuses_what_it_cannot_tune(toy_csv):
    lda = runner.invoke(app, ["evaluate", toy_csv, "--models", "lda", "--tune"])
    assert lda.exit_code == 1 and "default grid" in lda.output
    inner = runner.invoke(
        app, ["evaluate", toy_csv, "--models", "svm_rbf", "--tune", "--inner", "kfold"]
    )
    assert inner.exit_code == 1 and "--inner" in inner.output
    folds = runner.invoke(
        app, ["evaluate", toy_csv, "--models", "svm_rbf", "--tune", "--inner", "group_kfold",
              "--inner-splits", "1", "--permutations", "0"]
    )  # fmt: skip
    assert folds.exit_code == 1 and "inner_splits" in folds.output


def test_evaluate_without_permutations_skips_that_step(toy_csv):
    result = runner.invoke(app, ["evaluate", toy_csv, "--models", "lda", "--permutations", "0"])
    assert result.exit_code == 0 and "Permutation test" not in result.output


def test_evaluate_rejects_bad_input_with_a_clear_message(toy_csv):
    unknown_model = runner.invoke(app, ["evaluate", toy_csv, "--models", "lda,deep_net"])
    assert unknown_model.exit_code == 1 and "deep_net" in unknown_model.output
    missing = runner.invoke(app, ["evaluate", toy_csv, "--label", "nope"])
    assert missing.exit_code == 1 and "nope" in missing.output
    bad_norm = runner.invoke(app, ["evaluate", toy_csv, "--normalize", "robust"])
    assert bad_norm.exit_code == 1 and "--normalize" in bad_norm.output
    bad_log = runner.invoke(app, ["evaluate", toy_csv, "--log-features", "x"])
    assert bad_log.exit_code == 1 and "--log-features" in bad_log.output


def test_evaluate_refuses_a_table_with_errors(toy_csv, tmp_path):
    df = pd.read_csv(toy_csv)
    df.loc[0, "amp_1"] = np.nan
    path = tmp_path / "nan.csv"
    df.to_csv(path, index=False)
    result = runner.invoke(app, ["evaluate", str(path), "--permutations", "0"])
    assert result.exit_code == 1 and "non_finite" in result.output


def test_power_runs_from_a_pilot_table(toy_csv, tmp_path):
    out = tmp_path / "power"
    result = runner.invoke(
        app,
        ["power", toy_csv, "--log-features", "amp_1,amp_2,amp_3,amp_4", "--subjects", "4,6",
         "--trials", "2", "--reps", "6", "--effect", "0.3", "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Generator fitted on 6 pilot subjects" in result.output
    assert "False positives of the two classical tests" in result.output
    assert (out / "power.csv").exists() and (out / "power.png").exists()
    assert len(pd.read_csv(out / "power.csv")) == 2


@pytest.mark.parametrize(
    "args",
    [["--subjects", "1,4"], ["--trials", "0"], ["--model", "deep_net"], ["--x", "missing"]],
)
def test_power_rejects_invalid_designs(toy_csv, args):
    result = runner.invoke(app, ["power", toy_csv, "--reps", "2", *args])
    assert result.exit_code == 1
