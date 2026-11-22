"""Reading feature tables and choosing the feature columns.

The package works on one tidy table with one row per trial (or window): a subject column, a class
label, optionally a numeric covariate, and numeric feature columns. Column names follow the same
convention as `biosigkit <https://github.com/juanbezerra1/biosigkit>`_, whose feature tables can be
used as they are::

    subject, level, <channel>_<feature>, <channel>_<feature>, ...
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

#: Bookkeeping columns that are never features (same set as biosigkit, plus the covariate ``x``).
RESERVED_COLUMNS = frozenset(
    {
        "subject",
        "session",
        "trial",
        "segment_id",
        "kind",
        "t_start",
        "t_stop",
        "label",
        "level",
        "x",
    }
)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV file (comma or semicolon separated, decimal point) into a DataFrame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=None, engine="python")
    if df.empty:
        raise ValueError(f"{path} has no rows")
    return df


def infer_features(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """Numeric columns that are neither reserved bookkeeping columns nor in ``exclude``."""
    skip = set(RESERVED_COLUMNS) | set(exclude)
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def split_names(text: str | Sequence[str] | None) -> list[str]:
    """``"a, b,c"`` -> ``["a", "b", "c"]``; ``None`` or empty -> ``[]``."""
    if text is None:
        return []
    items = text.split(",") if isinstance(text, str) else list(text)
    return [item.strip() for item in items if item.strip()]


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a ``ValueError`` that lists every missing column."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"columns not found: {missing}; available: {list(df.columns)}")
