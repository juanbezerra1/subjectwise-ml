import numpy as np
import pandas as pd
import pytest

import subjectwise as sw
from subjectwise.correlation import choose_method, correlate


def _simpson(n_subjects: int = 8, rows: int = 12, seed: int = 0) -> pd.DataFrame:
    """Inside every subject y rises with x; across subjects, higher-x people have lower y."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_subjects):
        centre = 10.0 * s
        x = centre + rng.uniform(-3, 3, rows)
        y = 100.0 - 15.0 * s + 1.0 * (x - centre) + rng.normal(0, 0.5, rows)
        frames.append(pd.DataFrame({"subject": f"S{s}", "x": x, "y": y}))
    return pd.concat(frames, ignore_index=True)


def test_simpsons_paradox_pooled_and_within_subject_disagree_in_sign():
    rep = sw.correlation_report(_simpson(), "x", "y", "subject")
    assert rep.pooled.r < -0.5  # between-subject trend dominates
    assert rep.within.r > 0.8  # the real relationship inside each person
    assert rep.subject_level.r > 0.8
    assert rep.subject_level.p < 0.001
    assert rep.subject_level.n == 8
    assert len(rep.per_group) == 8
    assert (rep.per_group["rho"] > 0.5).all()


def test_subject_level_p_value_treats_subjects_not_rows_as_observations():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "subject": np.repeat(["A", "B", "C", "D"], 50),
            "x": rng.normal(size=200),
            "y": rng.normal(size=200),
        }
    )
    rep = sw.correlation_report(df, "x", "y", "subject")
    assert rep.subject_level.n == 4
    assert rep.subject_level.p > 0.05  # nothing is there; 200 rows do not make it significant


def test_subject_level_needs_three_subjects_with_enough_rows():
    df = _simpson(n_subjects=2)
    rep = sw.correlation_report(df, "x", "y", "subject")
    assert np.isnan(rep.subject_level.r)
    tiny = pd.DataFrame({"subject": ["A", "A", "B"], "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    per = sw.correlation_report(tiny, "x", "y", "subject").per_group
    assert per["rho"].isna().all()  # fewer than min_rows rows per subject


def test_log_x_requires_positive_values():
    df = _simpson().assign(x=lambda d: d["x"] - d["x"].min() + 1.0)
    sw.correlation_report(df, "x", "y", "subject", log_x=True)
    with pytest.raises(ValueError, match="strictly positive"):
        sw.correlation_report(df.assign(x=df["x"] - 100), "x", "y", "subject", log_x=True)


def test_correlate_basic_cases():
    x = np.arange(20.0)
    assert correlate(x, 2 * x + 1, method="pearson").r == pytest.approx(1.0)
    assert correlate(x, np.exp(x), method="spearman").r == pytest.approx(1.0)
    assert np.isnan(correlate(x, np.ones(20)).r)  # constant
    assert np.isnan(correlate(x[:2], x[:2]).r)  # too short
    with pytest.raises(ValueError, match="same length"):
        correlate(x, x[:5])
    with pytest.raises(ValueError, match="method"):
        correlate(x, x, method="kendall")


def test_choose_method_follows_normality():
    rng = np.random.default_rng(0)
    assert choose_method(rng.normal(size=100), rng.normal(size=100)) == "pearson"
    assert choose_method(rng.exponential(size=100), rng.normal(size=100)) == "spearman"
    assert choose_method(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == "spearman"
