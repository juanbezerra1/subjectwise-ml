import numpy as np
import pandas as pd
import pytest

import subjectwise as sw
from subjectwise.permutation import permute_within_groups


def test_shuffling_keeps_each_subjects_class_counts():
    rng = np.random.default_rng(0)
    groups = np.array(["A"] * 6 + ["B"] * 6, dtype=object)
    y = np.array([0, 0, 0, 1, 1, 2, 0, 1, 1, 1, 2, 2])
    out = permute_within_groups(y, groups, rng)
    for g in ("A", "B"):
        m = groups == g
        assert sorted(out[m]) == sorted(y[m])
    assert not np.array_equal(out, y)


def test_real_signal_gets_a_small_p_value(toy):
    df, feats, logs = toy
    res = sw.permutation_test(
        df,
        feats,
        "level",
        "subject",
        normalize="within_group",
        log_features=logs,
        n_perm=39,
        seed=1,
    )
    assert res.observed > 0.7
    assert res.null_mean == pytest.approx(0.25, abs=0.05)
    assert res.p_value == pytest.approx(1 / 40)  # no permutation reached the observed score
    assert res.summary()["n_perm"] == 39


def test_p_value_is_never_zero_and_follows_the_formula():
    res = sw.PermutationResult("mean_accuracy", observed=0.5, null=np.array([0.2, 0.5, 0.6, 0.3]))
    assert res.p_value == pytest.approx((1 + 2) / 5)  # null values >= observed: 0.5 and 0.6
    none = sw.PermutationResult("mean_accuracy", observed=0.9, null=np.zeros(9))
    assert none.p_value == pytest.approx(1 / 10)


def test_same_seed_same_null_and_independent_of_the_number_of_workers(toy):
    df, feats, logs = toy
    kwargs = dict(normalize="within_group", log_features=logs, n_perm=8, seed=5)
    a = sw.permutation_test(df, feats, "level", "subject", n_jobs=1, **kwargs)
    b = sw.permutation_test(df, feats, "level", "subject", n_jobs=1, **kwargs)
    c = sw.permutation_test(df, feats, "level", "subject", n_jobs=2, **kwargs)
    np.testing.assert_array_equal(a.null, b.null)
    np.testing.assert_array_equal(a.null, c.null)
    d = sw.permutation_test(df, feats, "level", "subject", **{**kwargs, "seed": 6})
    assert not np.array_equal(a.null, d.null)


def test_balanced_accuracy_metric_and_invalid_metric(toy):
    df, feats, logs = toy
    res = sw.permutation_test(
        df, feats, "level", "subject", normalize="within_group", log_features=logs,
        metric="balanced_accuracy", n_perm=5,
    )  # fmt: skip
    assert res.metric == "balanced_accuracy"
    with pytest.raises(ValueError, match="metric"):
        sw.permutation_test(df, feats, "level", "subject", metric="auc", n_perm=2)  # type: ignore[arg-type]


def test_false_positive_rate_is_not_inflated_on_pure_noise():
    """30 data sets without any signal: p <= 0.1 should happen about 10% of the time."""
    rejections = 0
    for seed in range(30):
        rng = np.random.default_rng(100 + seed)
        df = pd.DataFrame(
            {
                "subject": np.repeat([f"S{i}" for i in range(5)], 12),
                "y": np.tile(np.repeat([0, 1, 2], 4), 5),
                "f0": rng.normal(size=60),
                "f1": rng.normal(size=60),
            }
        )
        res = sw.permutation_test(df, ["f0", "f1"], "y", "subject", n_perm=19, seed=seed)
        rejections += res.p_value <= 0.1
    assert rejections <= 9  # expected 3; P(X >= 10) < 0.001 if the test were valid at 10%
