"""Nested tuning: settings are chosen inside the training subjects, never on the held-out one."""

import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import subjectwise as sw
from subjectwise import tuning as tn
from subjectwise.cv import group_accuracies
from subjectwise.models import make_model

SMALL = {"C": (1.0, 10.0), "gamma": (0.05, 0.5)}  # four candidates: fast, and the order matters


def noise_cohort(seed, n_subjects=6, rows=12, classes=3):
    """Features that say nothing about the label: whatever tuning finds is chance."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_subjects):
        frames.append(
            pd.DataFrame(
                {
                    "subject": f"S{i}",
                    "y": np.tile(np.arange(classes), rows // classes),
                    **{f"f{j}": rng.normal(size=rows) for j in range(3)},
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------------------ the grid
def test_candidates_vary_the_last_parameter_fastest_and_start_with_the_first_values():
    grid = {"a": (1, 2), "b": ("x", "y")}
    assert tn.candidates(grid) == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]


@pytest.mark.parametrize("name", sw.TUNABLE_MODELS)
def test_the_first_default_candidate_is_the_fixed_model(name):
    """So that, on a tie, tuning keeps the settings the untuned model would have used."""
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 4)), np.tile([0, 1, 2], 20)
    x[:, 0] += y  # something to learn
    first = tn.candidates(sw.default_grid(name, n_features=4))[0]
    tuned = make_model(name, 0, first).fit(x, y)
    fixed = make_model(name, 0).fit(x, y)
    assert np.array_equal(tuned.predict(x), fixed.predict(x))
    if name != "rf":
        assert np.allclose(tuned.decision_function(x), fixed.decision_function(x))


def test_gamma_in_the_default_grid_is_scikit_learns_scale_for_standardised_features():
    for d in (3, 12):
        assert sw.default_grid("svm_rbf", d)["gamma"][0] == pytest.approx(1 / d)
    assert set(sw.TUNABLE_MODELS) == {"svm_rbf", "svm_linear", "rf"}
    with pytest.raises(ValueError, match="no default grid"):
        sw.default_grid("lda", 4)


def test_grid_validation():
    ok = tn.make_tuning("svm_rbf", tune=True)
    assert ok == tn.Tuning(grid=None, inner="loso", inner_splits=5)
    assert tn.make_tuning("svm_rbf", tune=False) is None
    with pytest.raises(ValueError, match="only used with tune=True"):
        tn.make_tuning("svm_rbf", tune=False, grid=SMALL)
    with pytest.raises(ValueError, match="inner_scheme"):
        tn.make_tuning("svm_rbf", tune=True, inner_scheme="kfold")
    with pytest.raises(ValueError, match="inner_splits"):
        tn.make_tuning("svm_rbf", tune=True, inner_scheme="group_kfold", inner_splits=1)
    with pytest.raises(ValueError, match="at least one parameter"):
        tn.make_tuning("svm_rbf", tune=True, grid={})
    with pytest.raises(ValueError, match="non-empty list"):
        tn.make_tuning("svm_rbf", tune=True, grid={"C": []})
    with pytest.raises(ValueError, match="non-empty list"):
        tn.make_tuning("svm_rbf", tune=True, grid={"kernel": "rbf"})
    with pytest.raises(ValueError, match="no default grid"):
        tn.make_tuning("lda", tune=True)  # needs an explicit grid


def test_mean_group_accuracy_is_the_package_headline():
    y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0])
    pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
    groups = np.array(["a", "a", "a", 2, 2, 2, 2, "b", "b", "b"], dtype=object)
    assert tn.mean_group_accuracy(y, pred, groups) == pytest.approx(
        group_accuracies(y, pred, groups).mean()
    )


# --------------------------------------------------------------------------- choosing settings
def separable(n_subjects=4, rows=20, seed=0):
    """The class is the sign of f0; f1 is noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_subjects * rows, 2))
    y = (x[:, 0] > 0).astype(int)
    groups = np.repeat([f"S{i}" for i in range(n_subjects)], rows).astype(object)
    return x, y, groups


def test_the_candidate_with_the_best_inner_score_is_chosen():
    x, y, groups = separable()
    # gamma = 500 memorises the training rows and predicts one class for every new row
    grid = {"gamma": (500.0, 0.1)}
    choice = tn.select_params(x, y, groups, "svm_rbf", grid)
    assert choice.params == {"gamma": 0.1}
    assert choice.score > 0.85
    assert list(choice.table["gamma"]) == [500.0, 0.1]
    assert choice.table["inner_score"].iloc[0] < 0.65


def test_a_tie_goes_to_the_earliest_candidate():
    x, y, groups = separable()
    # both settings classify the separable data perfectly, so their scores are equal
    grid = {"C": (2.0, 1.0)}
    choice = tn.select_params(x, y, groups, "svm_linear", grid)
    assert choice.table["inner_score"].nunique() == 1
    assert choice.params == {"C": 2.0}
    first_wins = tn.select_params(x, y, groups, "svm_linear", {"C": (1.0, 2.0)})
    assert first_wins.params == {"C": 1.0}


def test_tuning_needs_at_least_two_training_subjects():
    x, y, groups = separable(n_subjects=1)
    with pytest.raises(ValueError, match="at least 2 training subjects"):
        tn.select_params(x, y, groups, "svm_rbf", SMALL)


def test_group_kfold_inner_loop_never_splits_a_subject():
    x, y, groups = separable(n_subjects=6)
    folds = tn._inner_splits("group_kfold", x, y, groups, 3)
    assert len(folds) == 3
    for train, test in folds:
        assert not set(groups[train].tolist()) & set(groups[test].tolist())
    assert len(tn._inner_splits("group_kfold", x, y, groups, 50)) == 6  # at most one per subject
    assert len(tn._inner_splits("loso", x, y, groups, 3)) == 6


# ------------------------------------------------------- the outer loop, against a plain loop
def plain_nested_loso(x, y, groups, grid, random_state=0):
    """The whole procedure written out with scikit-learn pieces and loops, nothing shared."""
    names = list(grid)
    pred = np.empty(len(y), dtype=y.dtype)
    rows = []
    for held in sorted(set(groups.tolist())):
        train = np.flatnonzero(groups != held)
        inner_subjects = sorted(set(groups[train].tolist()))
        scores = []
        for combo in itertools.product(*grid.values()):
            params = dict(zip(names, combo, strict=True))
            per_subject = []
            for inner in inner_subjects:
                fit_rows = train[groups[train] != inner]
                test_rows = train[groups[train] == inner]
                model = make_pipeline(
                    StandardScaler(), SVC(kernel="rbf", random_state=random_state, **params)
                ).fit(x[fit_rows], y[fit_rows])
                per_subject.append(np.mean(model.predict(x[test_rows]) == y[test_rows]))
            scores.append(float(np.mean(per_subject)))
        best = next(k for k, s in enumerate(scores) if s >= max(scores) - 1e-12)
        params = dict(zip(names, list(itertools.product(*grid.values()))[best], strict=True))
        final = make_pipeline(
            StandardScaler(), SVC(kernel="rbf", random_state=random_state, **params)
        ).fit(x[train], y[train])
        test = np.flatnonzero(groups == held)
        pred[test] = final.predict(x[test])
        rows.append({"held_out": held, "inner_score": scores[best], **params})
    return pred, pd.DataFrame(rows)


def test_evaluate_matches_a_plain_nested_loop(toy):
    df, feats, _ = toy
    result = sw.evaluate(df, feats, "level", "subject", model="svm_rbf", tune=True, grid=SMALL)
    x = df[feats].to_numpy(dtype=float)
    y, groups = df["level"].to_numpy(), df["subject"].to_numpy(dtype=object)
    pred, rows = plain_nested_loso(x, y, groups, SMALL)
    assert np.array_equal(result.y_pred, pred)
    table = result.tuning
    assert list(table["held_out"]) == list(rows["held_out"])
    assert np.allclose(table["inner_score"], rows["inner_score"])
    assert list(table["param_C"]) == list(rows["C"])
    assert list(table["param_gamma"]) == list(rows["gamma"])


def test_the_tuned_and_the_fixed_result_share_everything_but_the_settings(toy):
    df, feats, logs = toy
    common = dict(model="svm_rbf", normalize="within_group", log_features=logs)
    fixed = sw.evaluate(df, feats, "level", "subject", **common)
    tuned = sw.evaluate(df, feats, "level", "subject", tune=True, grid=SMALL, **common)
    assert np.array_equal(tuned.y_true, fixed.y_true) and np.array_equal(tuned.groups, fixed.groups)
    assert (tuned.scheme, tuned.model, tuned.normalize) == (
        fixed.scheme,
        fixed.model,
        fixed.normalize,
    )
    assert fixed.tuning is None and fixed.inner_accuracy is None and fixed.optimism is None


# ------------------------------------------------------------------------- no subject leakage
def test_the_inner_loop_only_ever_receives_the_training_subjects(toy, monkeypatch):
    df, feats, logs = toy
    seen = []
    real = tn.select_params

    def spy(x, y, groups, *args, **kwargs):
        seen.append((set(groups.tolist()), len(y)))
        return real(x, y, groups, *args, **kwargs)

    monkeypatch.setattr(tn, "select_params", spy)
    everyone = set(df["subject"])
    rows_per_subject = df.groupby("subject").size()

    sw.evaluate(df, feats, "level", "subject", model="svm_rbf", tune=True, grid=SMALL,
                log_features=logs)  # fmt: skip
    assert len(seen) == 6  # one inner loop per held-out subject, no more
    for (groups, n_rows), held_out in zip(seen, sorted(everyone), strict=True):
        assert groups == everyone - {held_out}
        assert n_rows == len(df) - rows_per_subject[held_out]

    seen.clear()
    result = sw.evaluate(df, feats, "level", "subject", model="svm_rbf", scheme="group_kfold",
                         n_splits=3, tune=True, grid=SMALL, log_features=logs)  # fmt: skip
    assert len(seen) == 3
    for (groups, _), held_out in zip(seen, result.tuning["held_out"], strict=True):
        left_out = set(held_out.split(", "))
        assert len(left_out) == 2 and not groups & left_out and groups | left_out == everyone


def test_what_the_held_out_subject_looks_like_cannot_change_what_was_chosen_for_it(toy):
    """Scramble one subject's features and labels: the choice made *for* them is the same."""
    df, feats, _ = toy
    kwargs = dict(model="svm_rbf", tune=True, grid=SMALL)
    base = sw.evaluate(df, feats, "level", "subject", **kwargs)
    mine = df["subject"] == "S03"
    rng = np.random.default_rng(1)
    scrambled = df.copy()
    scrambled.loc[mine, feats] = rng.normal(5.0, 20.0, size=(mine.sum(), len(feats)))
    scrambled.loc[mine, "level"] = rng.permutation(df.loc[mine, "level"].to_numpy())
    changed = sw.evaluate(scrambled, feats, "level", "subject", **kwargs)

    own = base.tuning["held_out"] == "S03"
    cols = ["param_C", "param_gamma", "inner_score"]
    pd.testing.assert_frame_equal(base.tuning.loc[own, cols], changed.tuning.loc[own, cols])
    # ...while the scrambled subject does reach the inner loops of everybody else
    assert not np.allclose(
        base.tuning.loc[~own, "inner_score"], changed.tuning.loc[~own, "inner_score"]
    )
    assert (
        base.tuning.loc[own, "outer_accuracy"].iloc[0]
        != changed.tuning.loc[own, "outer_accuracy"].iloc[0]
    )


def test_a_leak_inside_the_inner_loop_is_caught(monkeypatch):
    x, y, groups = separable(n_subjects=4)
    leaky = [(np.arange(len(y)), np.arange(len(y)))]
    monkeypatch.setattr(tn, "_inner_splits", lambda *args: leaky)
    with pytest.raises(AssertionError, match="leakage inside the tuning loop"):
        tn.select_params(x, y, groups, "svm_rbf", SMALL)


# ----------------------------------------------------------------------------- evaluate(...)
def test_the_tuning_table_has_one_row_per_outer_fold(toy):
    df, feats, logs = toy
    res = sw.evaluate(df, feats, "level", "subject", model="svm_rbf", tune=True, grid=SMALL,
                      normalize="within_group", log_features=logs)  # fmt: skip
    table = res.tuning
    assert list(table.columns) == [
        "fold", "held_out", "n_held_out", "n_train_subjects", "param_C", "param_gamma",
        "inner_score", "outer_accuracy",
    ]  # fmt: skip
    assert list(table["held_out"]) == [f"S0{i}" for i in range(1, 7)]
    assert (table["n_held_out"] == 1).all() and (table["n_train_subjects"] == 5).all()
    assert set(table["param_C"]) <= set(SMALL["C"]) and set(table["param_gamma"]) <= set(
        SMALL["gamma"]
    )
    # every row is predicted once, so the fold accuracies are the subjects' accuracies
    assert len(res.y_pred) == len(df) and not pd.isna(pd.Series(res.y_pred)).any()
    per_subject = res.per_group().set_index("group")["accuracy"]
    assert np.allclose(table["outer_accuracy"], per_subject.loc[table["held_out"]])
    assert res.mean_accuracy == pytest.approx(table["outer_accuracy"].mean())
    assert res.inner_accuracy == pytest.approx(table["inner_score"].mean())
    assert res.optimism == pytest.approx(res.inner_accuracy - res.mean_accuracy)


def test_summary_gains_the_inner_accuracy_only_when_tuned(toy):
    df, feats, logs = toy
    plain = sw.evaluate(df, feats, "level", "subject", model="svm_rbf", log_features=logs)
    assert set(plain.summary()) == {
        "mean_accuracy", "sd_accuracy", "pooled_accuracy", "balanced_accuracy", "f1_macro",
        "n_groups",
    }  # fmt: skip
    tuned = sw.evaluate(df, feats, "level", "subject", model="svm_rbf", tune=True, grid=SMALL,
                        log_features=logs)  # fmt: skip
    assert set(tuned.summary()) - set(plain.summary()) == {"inner_accuracy"}


def test_group_kfold_counts_every_held_out_subject_once(toy):
    df, feats, logs = toy
    res = sw.evaluate(df, feats, "level", "subject", model="svm_rbf", scheme="group_kfold",
                      n_splits=4, tune=True, grid=SMALL, inner_scheme="group_kfold",
                      inner_splits=2, log_features=logs)  # fmt: skip
    table = res.tuning
    assert table["n_held_out"].sum() == 6 and set(table["n_held_out"]) == {1, 2}
    weighted = np.average(table["inner_score"], weights=table["n_held_out"])
    assert res.inner_accuracy == pytest.approx(weighted)
    assert (table["n_train_subjects"] == 6 - table["n_held_out"]).all()


def test_the_same_seed_gives_the_same_choices(toy):
    df, feats, _ = toy
    a = sw.evaluate(df, feats, "level", "subject", model="rf", tune=True,
                    grid={"min_samples_leaf": (1, 5)}, random_state=3)  # fmt: skip
    b = sw.evaluate(df, feats, "level", "subject", model="rf", tune=True,
                    grid={"min_samples_leaf": (1, 5)}, random_state=3)  # fmt: skip
    pd.testing.assert_frame_equal(a.tuning, b.tuning)
    assert np.array_equal(a.y_pred, b.y_pred)


def test_the_random_forest_and_linear_svm_have_default_grids(toy):
    df, feats, logs = toy
    small = df[df["subject"].isin(["S01", "S02", "S03", "S04"])]
    for name, param in (("svm_linear", "param_C"), ("rf", "param_min_samples_leaf")):
        if name == "rf":  # the default random-forest grid is slow to run in full; narrow it
            res = sw.evaluate(small, feats, "level", "subject", model=name, tune=True,
                              grid={"min_samples_leaf": (1, 3), "max_features": ("sqrt", 0.5)},
                              log_features=logs)  # fmt: skip
        else:
            res = sw.evaluate(small, feats, "level", "subject", model=name, tune=True,
                              log_features=logs)  # fmt: skip
        assert param in res.tuning.columns and len(res.tuning) == 4


def test_any_model_can_be_tuned_with_an_explicit_grid(toy):
    df, feats, logs = toy
    res = sw.evaluate(df, feats, "level", "subject", model="lda", tune=True,
                      grid={"solver": ("svd", "lsqr")}, log_features=logs)  # fmt: skip
    assert set(res.tuning["param_solver"]) <= {"svd", "lsqr"}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"scheme": "kfold"}, "grouped scheme"),
        ({"model": "lda"}, "no default grid"),
        ({"model": "svm_rbf", "inner_scheme": "kfold"}, "inner_scheme"),
        ({"model": "svm_rbf", "grid": {"nonsense": (1, 2)}}, "nonsense"),
    ],
)
def test_evaluate_refuses_what_cannot_be_tuned_without_leakage(toy, kwargs, message):
    df, feats, _ = toy
    with pytest.raises(ValueError, match=message):
        sw.evaluate(df, feats, "level", "subject", tune=True, **{"model": "svm_rbf", **kwargs})


def test_a_grid_without_tune_is_an_error_not_silently_ignored(toy):
    df, feats, _ = toy
    with pytest.raises(ValueError, match="only used with tune=True"):
        sw.evaluate(df, feats, "level", "subject", model="svm_rbf", grid=SMALL)


def test_too_few_subjects_to_tune():
    df = noise_cohort(0, n_subjects=2)
    with pytest.raises(ValueError, match="at least 2 training subjects"):
        sw.evaluate(df, ["f0", "f1", "f2"], "y", "subject", model="svm_rbf", tune=True)
    with pytest.raises(ValueError, match="at least 3 subjects"):
        sw.permutation_test(df, ["f0", "f1", "f2"], "y", "subject", model="svm_rbf", tune=True,
                            n_perm=2)  # fmt: skip


def test_the_sensitivity_table_can_follow_the_tuned_procedure(toy):
    df, feats, logs = toy
    kwargs = dict(
        model="svm_rbf", normalize="within_group", log_features=logs, tune=True, grid=SMALL
    )
    table = sw.leave_group_out(df, feats, "level", "subject", groups=["S01"], **kwargs)
    assert list(table["scenario"]) == ["all", "without S01"]
    nested = sw.evaluate(df, feats, "level", "subject", **kwargs)
    assert table["mean_accuracy"].iloc[0] == pytest.approx(nested.mean_accuracy)


def test_the_optimism_is_visible_when_there_is_nothing_to_find():
    """Best of 12 candidates on noise: the inner score rises above chance, the outer does not."""
    inner, outer = [], []
    for seed in range(10):
        df = noise_cohort(seed)
        res = sw.evaluate(df, ["f0", "f1", "f2"], "y", "subject", model="svm_rbf", tune=True)
        inner.append(res.inner_accuracy)
        outer.append(res.mean_accuracy)
    assert np.mean(inner) - np.mean(outer) > 0.03
    assert abs(np.mean(outer) - 1 / 3) < 0.06  # the outer score is not inflated by the choice


# --------------------------------------------------------------------------- permutation test
def test_the_permutation_test_repeats_the_tuning_for_every_permutation(toy, monkeypatch):
    df, feats, logs = toy
    calls = []
    real = tn.select_params
    monkeypatch.setattr(tn, "select_params", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    res = sw.permutation_test(df, feats, "level", "subject", model="svm_rbf", tune=True,
                              grid=SMALL, log_features=logs, n_perm=3, seed=1)  # fmt: skip
    assert res.tuned and res.n_perm == 3
    assert len(calls) == (1 + 3) * 6  # (observed + permutations) x outer folds


def test_without_tuning_the_permutation_test_runs_no_inner_loop(toy, monkeypatch):
    df, feats, logs = toy

    def forbidden(*args, **kwargs):
        raise AssertionError("no tuning was asked for")

    monkeypatch.setattr(tn, "select_params", forbidden)
    res = sw.permutation_test(df, feats, "level", "subject", model="svm_rbf", log_features=logs,
                              n_perm=2, seed=1)  # fmt: skip
    assert not res.tuned


def test_the_tuned_observed_score_is_the_nested_loso_score(toy):
    df, feats, logs = toy
    kwargs = dict(
        model="svm_rbf", normalize="within_group", log_features=logs, tune=True, grid=SMALL
    )
    perm = sw.permutation_test(df, feats, "level", "subject", n_perm=2, seed=0, **kwargs)
    nested = sw.evaluate(df, feats, "level", "subject", **kwargs)
    assert perm.observed == pytest.approx(nested.mean_accuracy)


def test_the_tuned_null_is_reproducible_and_independent_of_the_number_of_workers(toy):
    df, feats, logs = toy
    kwargs = dict(model="svm_rbf", log_features=logs, tune=True, grid=SMALL, n_perm=4, seed=5)
    a = sw.permutation_test(df, feats, "level", "subject", n_jobs=1, **kwargs)
    b = sw.permutation_test(df, feats, "level", "subject", n_jobs=2, **kwargs)
    np.testing.assert_array_equal(a.null, b.null)
    assert a.observed == b.observed


def test_the_tuned_permutation_test_is_not_liberal_on_pure_noise():
    """40 data sets without any signal: p <= 0.1 should happen about 10% of the time."""
    hits = 0
    for seed in range(40):
        df = noise_cohort(seed, n_subjects=5)
        res = sw.permutation_test(df, ["f0", "f1", "f2"], "y", "subject", model="svm_linear",
                                  tune=True, grid={"C": (1.0, 0.1)}, n_perm=19,
                                  seed=seed)  # fmt: skip
        hits += res.p_value <= 0.1
    assert hits <= 10  # expected 4; P(X >= 11) is well below 1% if the test is valid at 10%
