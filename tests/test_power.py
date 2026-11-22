import warnings

import numpy as np
import pandas as pd
import pytest

import subjectwise as sw
from subjectwise.power import _one_sided_t


@pytest.fixture(scope="module")
def fitted():
    df = sw.toy_cohort(n_subjects=8, trials_per_level=5, seed=0)
    feats, logs = sw.toy_features()
    gen = sw.HierarchicalGenerator.fit(df, feats, "subject", "x", log_features=logs)
    return gen, sw.level_x_stats(df, "level", "x")


def test_level_x_stats_has_one_row_per_class():
    df = sw.toy_cohort(n_subjects=4, trials_per_level=4, seed=0)
    stats = sw.level_x_stats(df, "level", "x")
    assert list(stats.index) == [5, 20, 35, 50]
    assert list(stats.columns) == ["mean", "sd"]
    assert stats["mean"].iloc[0] == pytest.approx(5, abs=2)


def test_t_test_handles_identical_accuracies_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _one_sided_t(np.array([1.0, 1.0, 1.0]), 0.25) == 0.0
        assert _one_sided_t(np.array([0.25, 0.25]), 0.25) == 1.0
    assert np.isnan(_one_sided_t(np.array([0.9]), 0.25))
    assert _one_sided_t(np.array([0.9, 0.8, 0.85, 0.95]), 0.25) < 0.001


def test_strong_effect_is_detected_and_no_effect_is_not(fitted):
    gen, xs = fitted
    kwargs = dict(
        n_subjects_list=(8,), trials_per_level_list=(2,), n_reps=40, null_reps=100,
        normalize="within_group", seed=1,
    )  # fmt: skip
    strong = sw.power_curve(gen, xs, **kwargs).summary.iloc[0]
    null = sw.power_curve(gen.scale_effect(0.0), xs, **kwargs).summary.iloc[0]
    assert strong["power_calibrated"] > 0.9
    assert null["power_calibrated"] < 0.2  # 5% by construction, plus Monte-Carlo noise
    assert null["acc_median"] == pytest.approx(0.25, abs=0.05)
    assert strong["acc_median"] > 0.7


def test_summary_columns_ranges_and_design_grid(fitted):
    gen, xs = fitted
    res = sw.power_curve(
        gen.scale_effect(0.2), xs, n_subjects_list=(4, 6), trials_per_level_list=(2, 3),
        n_reps=10, null_reps=30, normalize="within_group", seed=2,
    )  # fmt: skip
    s = res.summary
    assert list(s.columns) == [
        "model", "trials_per_level", "n_subjects", "acc_median", "acc_p05", "acc_p95",
        "null_threshold", "power_calibrated", "power_ttest", "power_binom",
    ]  # fmt: skip
    assert len(s) == 4
    for col in ("power_calibrated", "power_ttest", "power_binom", "acc_median", "null_threshold"):
        assert s[col].between(0, 1).all()
    assert (s["acc_p05"] <= s["acc_median"]).all() and (s["acc_median"] <= s["acc_p95"]).all()
    assert len(res.runs) == 4 * 10 and len(res.null_runs) == 4 * 30


def test_false_positive_rates_report_both_tests(fitted):
    gen, xs = fitted
    res = sw.power_curve(
        gen, xs, n_subjects_list=(4,), trials_per_level_list=(2,), n_reps=5, null_reps=50,
        normalize="within_group", seed=3,
    )  # fmt: skip
    fpr = res.false_positive_rates()
    assert list(fpr.columns) == [
        "model",
        "trials_per_level",
        "n_subjects",
        "null_acc_median",
        "fpr_binom",
        "fpr_ttest",
        "n_null",
    ]
    assert fpr.loc[0, "n_null"] == 50
    assert fpr[["fpr_binom", "fpr_ttest"]].to_numpy().max() <= 0.4


def test_same_seed_same_result(fitted):
    gen, xs = fitted
    kwargs = dict(
        n_subjects_list=(4,),
        trials_per_level_list=(2,),
        n_reps=6,
        null_reps=20,
        normalize="within_group",
    )
    a = sw.power_curve(gen, xs, seed=7, **kwargs).runs
    b = sw.power_curve(gen, xs, seed=7, **kwargs).runs
    c = sw.power_curve(gen, xs, seed=8, **kwargs).runs
    pd.testing.assert_frame_equal(a, b)
    assert not a["mean_accuracy"].equals(c["mean_accuracy"])


def test_without_calibration_there_is_no_null_and_no_calibrated_power(fitted):
    gen, xs = fitted
    res = sw.power_curve(
        gen, xs, n_subjects_list=(4,), trials_per_level_list=(2,), n_reps=4, calibrate=False
    )
    assert res.null_runs.empty
    assert "power_calibrated" not in res.summary.columns
    with pytest.raises(ValueError, match="calibrate"):
        res.false_positive_rates()


def test_invalid_arguments(fitted):
    gen, xs = fitted
    with pytest.raises(ValueError, match="alpha"):
        sw.power_curve(gen, xs, alpha=1.5)
    with pytest.raises(ValueError, match="n_reps"):
        sw.power_curve(gen, xs, n_reps=0)
