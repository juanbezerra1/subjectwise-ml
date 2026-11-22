import numpy as np
import pandas as pd

import subjectwise as sw


def _codes(report: sw.AuditReport) -> set[str]:
    return {i.code for i in report.issues}


def test_clean_balanced_table_has_no_issues(toy):
    df, feats, _ = toy
    rep = sw.audit(df, feats, "level", "subject")
    assert rep.ok
    assert rep.issues == []
    assert rep.n_groups == 6 and rep.n_rows == len(df)
    assert rep.chance_rate == 0.25 and rep.majority_rate == 0.25
    assert "no issues found" in rep.format()


def test_non_finite_values_are_an_error(toy):
    df, feats, _ = toy
    bad = df.copy()
    bad.loc[3, feats[0]] = np.nan
    rep = sw.audit(bad, feats, "level", "subject")
    assert not rep.ok
    assert "non_finite" in _codes(rep)


def test_too_few_groups_and_single_class(toy):
    df, feats, _ = toy
    one = df[df["subject"] == "S01"]
    assert "too_few_groups" in _codes(sw.audit(one, feats, "level", "subject"))
    two = df[df["subject"].isin(["S01", "S02"])]
    rep = sw.audit(two, feats, "level", "subject")
    assert rep.ok and "two_groups" in _codes(rep)
    same = df.assign(level=5)
    assert "single_class" in _codes(sw.audit(same, feats, "level", "subject"))


def test_class_missing_in_one_subject_and_tiny_subject(toy):
    df, feats, _ = toy
    holes = df[~((df["subject"] == "S02") & (df["level"] == 50))]
    rep = sw.audit(holes, feats, "level", "subject")
    assert "missing_class_in_group" in _codes(rep)
    assert any("S02" in i.message for i in rep.issues if i.code == "missing_class_in_group")
    tiny = pd.concat([df, df[df["subject"] == "S01"].head(2).assign(subject="S99")])
    assert "tiny_group" in _codes(sw.audit(tiny, feats, "level", "subject"))


def test_imbalanced_classes_are_flagged_with_the_majority_baseline(toy):
    df, feats, _ = toy
    skewed = pd.concat([df, df[df["level"] == 5], df[df["level"] == 5]], ignore_index=True)
    rep = sw.audit(skewed, feats, "level", "subject")
    assert "imbalanced_classes" in _codes(rep)
    assert rep.majority_rate > 0.4


def test_copied_subject_is_detected_even_if_rows_are_reordered(toy):
    df, feats, _ = toy
    twin = df[df["subject"] == "S03"].sample(frac=1.0, random_state=0).assign(subject="S03_copy")
    rep = sw.audit(pd.concat([df, twin], ignore_index=True), feats, "level", "subject")
    dup = [i for i in rep.issues if i.code == "duplicate_groups"]
    assert len(dup) == 1
    assert "S03" in dup[0].message and "S03_copy" in dup[0].message


def test_signed_zero_does_not_hide_a_duplicate():
    base = pd.DataFrame(
        {"subject": ["A"] * 3, "level": [0, 1, 0], "f": [0.0, 1.0, 2.0], "g": [1.0, 2.0, 3.0]}
    )
    twin = base.assign(subject="B", f=[-0.0, 1.0, 2.0])
    rep = sw.audit(pd.concat([base, twin], ignore_index=True), ["f", "g"], "level", "subject")
    assert "duplicate_groups" in _codes(rep)


def test_constant_feature_is_reported(toy):
    df, feats, _ = toy
    rep = sw.audit(df.assign(flat=1.0), [*feats, "flat"], "level", "subject")
    assert "constant_feature" in _codes(rep)
    assert "[WARNING] constant_feature" in rep.format()
