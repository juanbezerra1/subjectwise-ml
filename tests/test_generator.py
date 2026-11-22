import numpy as np
import pandas as pd
import pytest

import subjectwise as sw


@pytest.fixture(scope="module")
def big_pilot():
    """40 subjects with a known truth (see toy_cohort): the fit should recover it."""
    df = sw.toy_cohort(n_subjects=40, trials_per_level=8, seed=11)
    feats, logs = sw.toy_features()
    gen = sw.HierarchicalGenerator.fit(df, feats, "subject", "x", log_features=logs, x_center=30.0)
    return df, feats, logs, gen


def test_fit_recovers_the_known_parameters(big_pilot):
    _, _, _, gen = big_pilot
    table = gen.parameter_table().set_index("feature")
    for j in range(
        4
    ):  # amp_j: log intercept 1 + 0.2 j, log slope 0.030 + 0.004 j, noise sd ~ 0.216
        row = table.loc[f"amp_{j + 1}"]
        assert row["scale"] == "log"
        assert row["intercept_mean"] == pytest.approx(1.0 + 0.2 * j, abs=0.25)
        assert row["slope_mean"] == pytest.approx(0.030 + 0.004 * j, abs=0.005)
        assert row["residual_sd"] == pytest.approx(np.hypot(0.12, 0.18), abs=0.02)
        assert 0.35 < row["intercept_sd_between"] < 0.7  # shared subject gain, sd 0.5
    spec = table.loc["spec_1"]
    assert spec["scale"] == "linear"
    assert spec["slope_mean"] == pytest.approx(-0.06, abs=0.02)
    assert spec["residual_sd"] == pytest.approx(3.0, abs=0.3)


def test_residuals_keep_the_correlation_between_features(big_pilot):
    _, _, _, gen = big_pilot
    sd = np.sqrt(np.diag(gen.resid_cov))
    corr = gen.resid_cov / np.outer(sd, sd)
    # amp_1 and amp_2 share a 0.12-sd fluctuation, so rho = 0.12**2 / (0.12**2 + 0.18**2) = 0.31
    assert 0.2 < corr[0, 1] < 0.42
    assert abs(corr[0, 4]) < 0.15  # amplitude vs spectral: independent


def test_simulate_cohort_shape_columns_and_reproducibility(big_pilot):
    _, feats, _, gen = big_pilot
    x = np.array([5.0, 20.0, 35.0, 50.0])
    labels = np.array([5, 20, 35, 50])
    a = gen.simulate_cohort([x, x, x], [labels] * 3, np.random.default_rng(0))
    b = gen.simulate_cohort([x, x, x], [labels] * 3, np.random.default_rng(0))
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["subject", "label", "x", *feats]
    assert len(a) == 12 and list(a["subject"].unique()) == ["SIM01", "SIM02", "SIM03"]
    assert (a[["amp_1", "amp_2", "amp_3", "amp_4"]] > 0).all().all()  # log features stay positive


def test_null_generator_removes_the_dependence_on_x(big_pilot):
    _, _, _, gen = big_pilot
    null = gen.scale_effect(0.0)
    f = gen.n_features
    assert (null.mu[f:] == 0).all() and (null.deviations[:, f:] == 0).all()
    rng = np.random.default_rng(0)
    a, b = null.sample_theta(rng)
    x = np.linspace(0, 60, 600)
    values = null.simulate_model_scale(a, b, x, rng)
    assert all(abs(np.corrcoef(x, values[:, j])[0, 1]) < 0.15 for j in range(f))
    # the original generator does show the dependence, and scaling is linear
    a1, b1 = gen.sample_theta(np.random.default_rng(1))
    strong = gen.simulate_model_scale(a1, b1, x, np.random.default_rng(1))
    assert np.corrcoef(x, strong[:, 0])[0, 1] > 0.3
    assert gen.scale_effect(2.0).mu[f:] == pytest.approx(2 * gen.mu[f:])


def test_variance_uncertainty_widens_the_simulated_population():
    df = sw.toy_cohort(n_subjects=6, trials_per_level=4, seed=2)
    feats, logs = sw.toy_features()
    wide = sw.HierarchicalGenerator.fit(df, feats, "subject", "x", log_features=logs)
    tight = sw.HierarchicalGenerator.fit(
        df, feats, "subject", "x", log_features=logs, variance_uncertainty=False
    )
    a_wide, _ = wide.sample_theta(np.random.default_rng(0), size=4000)
    a_tight, _ = tight.sample_theta(np.random.default_rng(0), size=4000)
    assert a_wide.std(axis=0).mean() > 1.05 * a_tight.std(axis=0).mean()


def test_spread_interval_is_wider_with_fewer_pilot_subjects():
    feats, logs = sw.toy_features()
    few = sw.HierarchicalGenerator.fit(
        sw.toy_cohort(5, trials_per_level=4, seed=0), feats, "subject", "x", log_features=logs
    )
    many = sw.HierarchicalGenerator.fit(
        sw.toy_cohort(20, trials_per_level=4, seed=0), feats, "subject", "x", log_features=logs
    )
    lo_f, hi_f = few.spread_interval()
    lo_m, hi_m = many.spread_interval()
    assert lo_f < 1.0 < hi_f and lo_m < 1.0 < hi_m
    assert (hi_f - lo_f) > (hi_m - lo_m)


def test_fit_validations_and_warnings():
    df = sw.toy_cohort(6, trials_per_level=4, seed=0)
    feats, logs = sw.toy_features()
    with pytest.raises(ValueError, match="log_features not in features"):
        sw.HierarchicalGenerator.fit(df, feats, "subject", "x", log_features=["nope"])
    one = df[df["subject"] == "S01"]
    with pytest.raises(ValueError, match="at least 2"):
        sw.HierarchicalGenerator.fit(one, feats, "subject", "x", log_features=logs)
    flat = df.copy()
    flat.loc[flat["subject"] == "S02", "x"] = 10.0  # no slope can be estimated for S02
    with pytest.warns(UserWarning, match="skipped"):
        gen = sw.HierarchicalGenerator.fit(flat, feats, "subject", "x", log_features=logs)
    assert gen.n_subjects == 5
    three = df[df["subject"].isin(["S01", "S02", "S03"])]
    with pytest.warns(UserWarning, match="pilot subjects"):
        sw.HierarchicalGenerator.fit(three, feats, "subject", "x", log_features=logs)


def test_calibration_check_is_roughly_calibrated_on_data_from_the_same_process(big_pilot):
    df, feats, logs, _ = big_pilot
    sub = df[df["subject"].isin([f"S{i:02d}" for i in range(1, 21)])]
    cal = sw.calibration_check(sub, feats, "subject", "x", log_features=logs, n_sim=200, seed=0)
    assert len(cal) == 20 and (cal["k_train"] == 19).all()
    values = cal[feats].to_numpy()
    assert ((values >= 0) & (values <= 100)).all()
    assert sw.fraction_calibrated(cal) > 0.75  # about 0.90 expected
    again = sw.calibration_check(sub, feats, "subject", "x", log_features=logs, n_sim=200, seed=0)
    pd.testing.assert_frame_equal(cal, again)


def test_calibration_check_flags_an_outlier_subject(big_pilot):
    df, feats, logs, _ = big_pilot
    sub = df[df["subject"].isin([f"S{i:02d}" for i in range(1, 11)])].copy()
    sub.loc[sub["subject"] == "S01", feats[:4]] *= 100.0  # one subject with 100x the gain
    cal = sw.calibration_check(sub, feats, "subject", "x", log_features=logs, n_sim=200, seed=0)
    row = cal[cal["held_out"] == "S01"].iloc[0]
    assert row["amp_1"] > 99  # far above every simulated subject
    assert sw.fraction_calibrated(
        pd.DataFrame(columns=["held_out", "k_train", *feats])
    ) != sw.fraction_calibrated(cal)


def test_fraction_calibrated_of_an_empty_table_is_nan():
    empty = pd.DataFrame(columns=["held_out", "k_train", "f"])
    assert np.isnan(sw.fraction_calibrated(empty))
