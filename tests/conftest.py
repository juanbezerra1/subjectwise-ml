"""Shared fixtures: small synthetic cohorts with a known structure."""

from __future__ import annotations

import pandas as pd
import pytest

import subjectwise as sw


@pytest.fixture(scope="session")
def toy() -> tuple[pd.DataFrame, list[str], list[str]]:
    """6 subjects x 16 trials, 4 classes; ``(table, features, log_features)``."""
    df = sw.toy_cohort(n_subjects=6, trials_per_level=4, seed=3)
    features, log_features = sw.toy_features()
    return df, features, log_features


@pytest.fixture(scope="session")
def toy_csv(tmp_path_factory: pytest.TempPathFactory, toy) -> str:
    df, _, _ = toy
    path = tmp_path_factory.mktemp("data") / "toy.csv"
    df.to_csv(path, index=False)
    return str(path)
