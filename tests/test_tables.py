import pandas as pd
import pytest

import subjectwise as sw
from subjectwise.tables import require_columns, split_names


def test_infer_features_skips_bookkeeping_and_non_numeric_columns():
    df = pd.DataFrame(
        {
            "subject": ["a"], "session": [1], "trial": [1], "level": [5], "x": [1.0],
            "t_start": [0.0], "emg1_rms": [1.0], "note": ["text"], "emg1_mdf": [60.0],
        }
    )  # fmt: skip
    assert sw.infer_features(df) == ["emg1_rms", "emg1_mdf"]
    assert sw.infer_features(df, exclude=["emg1_mdf"]) == ["emg1_rms"]


def test_reserved_columns_include_the_biosigkit_bookkeeping_set():
    expected = {
        "subject",
        "session",
        "trial",
        "segment_id",
        "kind",
        "t_start",
        "t_stop",
        "label",
        "level",
    }
    assert expected <= sw.RESERVED_COLUMNS


def test_read_table_accepts_comma_and_semicolon_separators(tmp_path):
    (tmp_path / "a.csv").write_text("subject,level,f\nS1,5,1.5\nS2,20,2.5\n")
    (tmp_path / "b.csv").write_text("subject;level;f\nS1;5;1.5\nS2;20;2.5\n")
    a, b = sw.read_table(tmp_path / "a.csv"), sw.read_table(tmp_path / "b.csv")
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["subject", "level", "f"]


def test_read_table_rejects_missing_and_empty_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        sw.read_table(tmp_path / "nope.csv")
    (tmp_path / "empty.csv").write_text("subject,level,f\n")
    with pytest.raises(ValueError, match="no rows"):
        sw.read_table(tmp_path / "empty.csv")


def test_split_names_and_require_columns():
    assert split_names("a, b,,c ") == ["a", "b", "c"]
    assert split_names(None) == []
    assert split_names(["x", " y "]) == ["x", "y"]
    with pytest.raises(ValueError, match=r"\['z'\]"):
        require_columns(pd.DataFrame({"a": [1]}), ["a", "z"])
