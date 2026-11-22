import numpy as np
import pandas as pd
import pytest

import subjectwise as sw
from subjectwise.cv import EvalResult, group_accuracies, prepare


def _subject_fingerprint_cohort(n_subjects: int = 8, rows: int = 20, seed: int = 0) -> pd.DataFrame:
    """Features identify the subject, never the class; each subject mostly shows one class."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_subjects):
        majority = s % 2
        labels = np.where(rng.random(rows) < 0.9, majority, 1 - majority)
        offset = rng.normal(0.0, 5.0, size=3)
        frames.append(
            pd.DataFrame(
                {
                    "subject": f"S{s}",
                    "y": labels,
                    **{f"f{j}": offset[j] + rng.normal(0, 0.3, rows) for j in range(3)},
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_random_split_is_optimistic_when_features_identify_the_subject():
    df = _subject_fingerprint_cohort()
    feats = ["f0", "f1", "f2"]
    naive = sw.evaluate(df, feats, "y", "subject", model="svm_rbf", scheme="kfold")
    honest = sw.evaluate(df, feats, "y", "subject", model="svm_rbf", scheme="loso")
    assert naive.pooled_accuracy > 0.8  # the model recognises the person
    assert honest.mean_accuracy < 0.7  # a new person: nothing to recognise
    assert naive.pooled_accuracy - honest.mean_accuracy > 0.2


def test_grouped_schemes_never_share_a_subject_between_train_and_test():
    df = _subject_fingerprint_cohort()
    for scheme in ("loso", "group_kfold"):
        # evaluate() raises AssertionError itself if a subject lands on both sides
        res = sw.evaluate(df, ["f0", "f1", "f2"], "y", "subject", scheme=scheme, n_splits=3)
        assert len(res.y_pred) == len(df)


def test_every_row_is_predicted_exactly_once(toy):
    df, feats, logs = toy
    for scheme in ("loso", "group_kfold", "kfold"):
        res = sw.evaluate(df, feats, "level", "subject", scheme=scheme, log_features=logs)
        assert not pd.isna(pd.Series(res.y_pred)).any()
        assert len(res.y_pred) == len(df)


def test_mean_accuracy_counts_every_subject_once():
    # subject A: 10 rows, 100% correct; subject B: 2 rows, 0% correct
    y_true = np.array([1] * 12)
    y_pred = np.array([1] * 10 + [0] * 2)
    groups = np.array(["A"] * 10 + ["B"] * 2, dtype=object)
    res = EvalResult("loso", "lda", "none", y_true, y_pred, groups)
    assert res.mean_accuracy == pytest.approx(0.5)
    assert res.pooled_accuracy == pytest.approx(10 / 12)
    assert res.sd_accuracy == pytest.approx(np.std([1.0, 0.0], ddof=1))
    np.testing.assert_allclose(group_accuracies(y_true, y_pred, groups), [1.0, 0.0])


def test_single_group_has_zero_sd():
    res = EvalResult(
        "loso",
        "lda",
        "none",
        np.array([1, 0]),
        np.array([1, 1]),
        np.array(["A", "A"], dtype=object),
    )
    assert res.sd_accuracy == 0.0


def test_string_integer_and_whole_float_labels_all_work(toy):
    df, feats, logs = toy
    base = sw.evaluate(df, feats, "level", "subject", log_features=logs).mean_accuracy
    as_text = df.assign(level=df["level"].map({5: "low", 20: "mid", 35: "high", 50: "max"}))
    as_float = df.assign(level=df["level"].astype(float))
    assert sw.evaluate(
        as_text, feats, "level", "subject", log_features=logs
    ).mean_accuracy == pytest.approx(base)
    assert sw.evaluate(
        as_float, feats, "level", "subject", log_features=logs
    ).mean_accuracy == pytest.approx(base)


def test_continuous_or_missing_labels_are_rejected(toy):
    df, feats, _ = toy
    with pytest.raises(ValueError, match="discrete"):
        sw.evaluate(df.assign(level=df["x"]), feats, "level", "subject")
    missing = df.copy()
    missing["level"] = missing["level"].astype(float)
    missing.loc[0, "level"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        sw.evaluate(missing, feats, "level", "subject")


def test_confusion_matrix_orders_numeric_labels_numerically(toy):
    df, feats, logs = toy
    res = sw.evaluate(df, feats, "level", "subject", normalize="within_group", log_features=logs)
    assert list(res.confusion().index) == ["5", "20", "35", "50"]  # not '20', '35', '5', '50'
    assert res.confusion().to_numpy().sum() == len(df)
    assert res.labels == ["5", "20", "35", "50"]


def test_prepare_rejects_non_finite_features_and_unknown_normalisation(toy):
    df, feats, _ = toy
    broken = df.copy()
    broken.loc[0, feats[0]] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        prepare(broken, feats, "level", "subject")
    with pytest.raises(ValueError, match="normalize"):
        prepare(df, feats, "level", "subject", normalize="robust")


def test_loso_needs_two_subjects_and_unknown_scheme_is_rejected(toy):
    df, feats, _ = toy
    one = df[df["subject"] == "S01"]
    with pytest.raises(ValueError, match="at least 2"):
        sw.evaluate(one, feats, "level", "subject")
    with pytest.raises(ValueError, match="scheme"):
        sw.evaluate(df, feats, "level", "subject", scheme="bootstrap")  # type: ignore[arg-type]


def test_group_kfold_caps_the_number_of_folds_at_the_number_of_subjects(toy):
    df, feats, logs = toy
    res = sw.evaluate(
        df, feats, "level", "subject", scheme="group_kfold", n_splits=50, log_features=logs
    )
    assert len(res.y_pred) == len(df)


def test_compare_schemes_and_leakage_gap(toy):
    df, feats, logs = toy
    table = sw.compare_schemes(df, feats, "level", "subject", models=("lda",), log_features=logs)
    assert len(table) == len(sw.DEFAULT_SCENARIOS)
    assert {
        "model",
        "scheme",
        "normalize",
        "mean_accuracy",
        "sd_accuracy",
        "balanced_accuracy",
    } <= set(table.columns)
    gap = sw.leakage_gap(table)
    assert list(gap.columns) == ["model", "kfold_accuracy", "loso_accuracy", "gap_points"]
    row = gap.iloc[0]
    assert row["gap_points"] == pytest.approx((row["kfold_accuracy"] - row["loso_accuracy"]) * 100)


def test_within_subject_normalisation_helps_when_subjects_differ_in_gain(toy):
    df, feats, logs = toy
    raw = sw.evaluate(df, feats, "level", "subject", model="lda", log_features=logs)
    normed = sw.evaluate(
        df, feats, "level", "subject", model="lda", normalize="within_group", log_features=logs
    )
    assert normed.mean_accuracy > raw.mean_accuracy + 0.1


@pytest.mark.parametrize("model", sw.MODEL_NAMES)
def test_every_model_runs(toy, model):
    df, feats, logs = toy
    res = sw.evaluate(
        df, feats, "level", "subject", model=model, normalize="within_group", log_features=logs
    )
    assert 0.0 <= res.mean_accuracy <= 1.0
    assert set(res.summary()) >= {
        "mean_accuracy",
        "sd_accuracy",
        "pooled_accuracy",
        "balanced_accuracy",
        "f1_macro",
    }


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        sw.make_model("deep_net")
