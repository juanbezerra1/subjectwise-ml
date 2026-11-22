import pandas as pd
import pytest

import subjectwise as sw


def test_one_row_for_everybody_and_one_per_excluded_subject(toy):
    df, feats, logs = toy
    out = sw.leave_group_out(df, feats, "level", "subject", log_features=logs)
    assert len(out) == 1 + df["subject"].nunique()
    assert out.loc[0, "scenario"] == "all" and out.loc[0, "delta_points"] == 0.0
    assert (out["n_groups"].iloc[1:] == df["subject"].nunique() - 1).all()
    drop = out.iloc[1]
    assert drop["delta_points"] == pytest.approx(
        (drop["mean_accuracy"] - out.loc[0, "mean_accuracy"]) * 100
    )


def test_restricting_the_subjects_to_drop(toy):
    df, feats, logs = toy
    out = sw.leave_group_out(df, feats, "level", "subject", log_features=logs, groups=["S02"])
    assert list(out["dropped"]) == ["", "S02"]
    assert out["n_rows"].iloc[1] == (df["subject"] != "S02").sum()


def test_the_input_table_is_left_untouched(toy):
    df, feats, logs = toy
    before = df.copy()
    sw.leave_group_out(df, feats, "level", "subject", log_features=logs, groups=["S01"])
    pd.testing.assert_frame_equal(df, before)
