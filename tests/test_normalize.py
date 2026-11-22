import numpy as np
import pandas as pd
import pytest

import subjectwise as sw


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "subject": np.repeat(["A", "B", "C"], 10),
            "f1": np.concatenate(
                [rng.normal(5, 2, 10), rng.normal(50, 9, 10), rng.normal(-3, 1, 10)]
            ),
            "f2": rng.lognormal(0.0, 0.5, 30),
        }
    )


def test_each_group_has_zero_mean_and_unit_sample_sd():
    z = sw.within_group_zscore(_frame(), ["f1", "f2"], "subject")
    grouped = z.groupby("subject")[["f1", "f2"]]
    np.testing.assert_allclose(grouped.mean().to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(grouped.std(ddof=1).to_numpy(), 1.0, atol=1e-12)


def test_invariant_to_a_per_subject_affine_change_of_scale():
    df = _frame()
    changed = df.copy()
    mask = changed["subject"] == "B"
    changed.loc[mask, "f1"] = changed.loc[mask, "f1"] * 7.0 + 3.0
    a = sw.within_group_zscore(df, ["f1"], "subject")
    b = sw.within_group_zscore(changed, ["f1"], "subject")
    np.testing.assert_allclose(a["f1"], b["f1"], atol=1e-10)


def test_log_features_make_multiplicative_gain_irrelevant():
    df = _frame()
    changed = df.copy()
    mask = changed["subject"] == "C"
    changed.loc[mask, "f2"] *= 40.0  # a different electrode gain
    a = sw.within_group_zscore(df, ["f2"], "subject", log_features=["f2"])
    b = sw.within_group_zscore(changed, ["f2"], "subject", log_features=["f2"])
    np.testing.assert_allclose(a["f2"], b["f2"], atol=1e-10)


def test_does_not_modify_the_input_and_keeps_other_columns():
    df = _frame()
    before = df.copy()
    out = sw.within_group_zscore(df, ["f1"], "subject")
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_series_equal(out["f2"], df["f2"])


def test_constant_feature_and_single_row_group_become_zero_and_nan_is_kept():
    df = pd.DataFrame(
        {
            "subject": ["A", "A", "A", "B"],
            "const": [2.0, 2.0, 2.0, 5.0],
            "f": [1.0, 2.0, np.nan, 9.0],
        }
    )
    z = sw.within_group_zscore(df, ["const", "f"], "subject")
    assert (z["const"] == 0).all()
    assert z.loc[3, "f"] == 0.0  # single-row group
    assert np.isnan(z.loc[2, "f"])  # NaN stays NaN


def test_zeros_are_clipped_before_the_log_and_negatives_are_rejected():
    df = pd.DataFrame({"subject": ["A", "A", "A", "B", "B"], "p": [0.0, 1.0, 2.0, 3.0, 4.0]})
    z = sw.within_group_zscore(df, ["p"], "subject", log_features=["p"])
    assert np.isfinite(z["p"]).all()
    bad = df.assign(p=[-1.0, 1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="negative"):
        sw.within_group_zscore(bad, ["p"], "subject", log_features=["p"])


def test_unknown_log_feature_and_missing_group_are_rejected():
    df = _frame()
    with pytest.raises(ValueError, match="not in features"):
        sw.within_group_zscore(df, ["f1"], "subject", log_features=["f2"])
    df.loc[0, "subject"] = None
    with pytest.raises(ValueError, match="missing"):
        sw.within_group_zscore(df, ["f1"], "subject")
