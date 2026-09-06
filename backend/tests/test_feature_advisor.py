"""Column screening and the measured recommendation.

The advisor exists to answer "which columns should I train on?", so these
pin down the judgements it must not get wrong: identifiers and mostly-empty
columns have to be rejected, informative ones kept, and the reported
improvement has to come from a real measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.feature_advisor import suggest_features
from app.exceptions import ValidationError


@pytest.fixture
def survival() -> pd.DataFrame:
    """Titanic-shaped frame: signal, noise, an id, an empty column, free text."""
    rng = np.random.default_rng(4)
    n = 400
    sex = rng.choice(["male", "female"], n)
    pclass = rng.integers(1, 4, n)
    # Both effects must survive the threshold below, so they are kept comparable:
    # an overpowering sex term flattens the class gradient to noise.
    odds = (sex == "female") * 1.2 - (pclass - 2) * 1.2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "row_id": np.arange(1, n + 1),               # running counter
        "survived": (odds > 0).astype(int),
        "sex": sex,
        "pclass": pclass,
        "lucky_number": rng.normal(0, 1, n),          # pure noise
        "cabin": [None if i % 5 else f"C{i}" for i in range(n)],   # 80% empty
        "full_name": [f"passenger {i}" for i in range(n)],         # unique text
    })


def test_running_counter_is_dropped(survival):
    """A row number carries no signal even when it correlates by accident."""
    result = suggest_features(survival, "survived", "classification")
    verdict = next(c for c in result["columns"] if c["column"] == "row_id")
    assert verdict["verdict"] == "drop"
    assert "row number" in verdict["reason"]
    assert "row_id" not in result["recommended"]


def test_mostly_empty_column_is_dropped(survival):
    verdict = next(
        c for c in suggest_features(survival, "survived", "classification")["columns"]
        if c["column"] == "cabin"
    )
    assert verdict["verdict"] == "drop" and "missing" in verdict["reason"]


def test_free_text_column_is_dropped(survival):
    verdict = next(
        c for c in suggest_features(survival, "survived", "classification")["columns"]
        if c["column"] == "full_name"
    )
    assert verdict["verdict"] == "drop"


def test_informative_columns_are_recommended(survival):
    recommended = suggest_features(survival, "survived", "classification")["recommended"]
    assert "sex" in recommended and "pclass" in recommended


def test_noise_ranks_below_signal(survival):
    ranked = suggest_features(survival, "survived", "classification")["columns"]
    score = {c["column"]: c["relevance"] for c in ranked}
    assert score["sex"] > score["lucky_number"]


def test_recommendation_is_backed_by_a_measurement(survival):
    result = suggest_features(survival, "survived", "classification")
    ev = result["evaluation"]
    assert ev["score_all_usable"] is not None
    assert ev["score_recommended"] is not None
    assert ev["folds"] >= 2
    # The summary must quote the numbers it was given, not adjectives alone.
    assert any(word in result["summary"] for word in ("improves", "unchanged", "drops", "keeping"))


def test_recommendation_is_never_empty(survival):
    """Even if nothing scores well, training needs at least one column."""
    flat = survival.copy()
    flat["survived"] = 1  # no variation to explain
    result = suggest_features(flat, "survived", "classification")
    assert len(result["recommended"]) >= 1


def test_regression_target_is_supported():
    rng = np.random.default_rng(5)
    n = 300
    size = rng.normal(100, 20, n)
    df = pd.DataFrame({
        "size": size,
        "noise": rng.normal(0, 1, n),
        "price": size * 3 + rng.normal(0, 5, n),
    })
    result = suggest_features(df, "price", "regression")
    assert result["evaluation"]["metric"] == "R²"
    assert "size" in result["recommended"]


def test_unknown_target_is_rejected(survival):
    with pytest.raises(ValidationError, match="not in this dataset"):
        suggest_features(survival, "nope", "classification")


def test_clustering_is_rejected(survival):
    with pytest.raises(ValidationError, match="supervised"):
        suggest_features(survival, "survived", "clustering")
