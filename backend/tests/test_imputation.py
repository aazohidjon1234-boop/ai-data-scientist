"""Choosing how missing values are filled — and measuring whether it matters."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.improver import suggest_improvements
from app.exceptions import ValidationError
from app.tools.data_tools import clean_dataset, prepare_features


@pytest.fixture
def gappy() -> pd.DataFrame:
    """One extreme value, so median and mean disagree visibly."""
    return pd.DataFrame({
        "n": [1.0, 2.0, np.nan, 100.0],
        "c": ["a", "a", None, "b"],
    })


def test_median_is_the_default_and_resists_the_outlier(gappy):
    out, report = clean_dataset(gappy)
    assert out["n"].tolist() == [1.0, 2.0, 2.0, 100.0]
    assert report["imputed"]["n"]["strategy"] == "median"


def test_mean_is_dragged_by_the_outlier(gappy):
    out, _ = clean_dataset(gappy, numeric_strategy="mean")
    assert out["n"].iloc[2] == pytest.approx((1 + 2 + 100) / 3)


def test_zero_fills_with_zero(gappy):
    out, _ = clean_dataset(gappy, numeric_strategy="zero")
    assert out["n"].iloc[2] == 0.0


def test_mode_fills_the_most_common_category(gappy):
    out, _ = clean_dataset(gappy)
    assert out["c"].iloc[2] == "a"


def test_constant_does_not_invent_an_observed_value(gappy):
    out, report = clean_dataset(gappy, categorical_strategy="constant")
    assert out["c"].iloc[2] == "missing"
    assert report["imputed"]["c"]["strategy"] == "constant"


def test_unknown_strategy_is_rejected(gappy):
    with pytest.raises(ValidationError, match="numeric imputation"):
        clean_dataset(gappy, numeric_strategy="magic")
    with pytest.raises(ValidationError, match="categorical imputation"):
        clean_dataset(gappy, categorical_strategy="magic")


def test_feature_prep_honours_the_strategy_and_records_it():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 50.0], "y": [0, 1, 0, 1]})
    _, _, report = prepare_features(df, "y", "classification", numeric_strategy="zero")
    assert report["imputation"] == {"numeric": "zero", "categorical": "mode"}


def test_strategy_reaches_training(client, regression_dataset_id):
    r = client.post(f"/api/datasets/{regression_dataset_id}/train",
                    json={"target": "price", "impute_numeric": "mean"})
    assert r.status_code == 200, r.text


def _with_gaps() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n = 400
    age, bp = rng.normal(50, 12, n), rng.normal(130, 18, n)
    df = pd.DataFrame({
        "age": age, "bp": bp,
        "grp": rng.choice(["a", "b", "c"], n),
        "target": ((age - 50) * 0.10 + (bp - 130) * 0.05 + rng.normal(0, 0.7, n) > 0).astype(int),
    })
    df.loc[rng.choice(n, 80, replace=False), "bp"] = np.nan
    df.loc[rng.choice(n, 50, replace=False), "grp"] = None
    return df


def test_filling_options_are_offered_when_data_is_missing():
    keys = {r["key"] for r in suggest_improvements(_with_gaps(), "target", "classification")["recipes"]}
    assert "impute_mean" in keys and "impute_zero" in keys and "impute_constant" in keys


def test_filling_options_are_hidden_when_nothing_is_missing():
    """A complete dataset must not be offered a change that cannot do anything."""
    complete = _with_gaps().dropna()
    keys = {r["key"] for r in suggest_improvements(complete, "target", "classification")["recipes"]}
    assert not any(k.startswith("impute_") for k in keys)


def test_the_current_strategy_is_not_offered_back():
    result = suggest_improvements(_with_gaps(), "target", "classification",
                                  current_impute_numeric="mean")
    keys = {r["key"] for r in result["recipes"]}
    assert "impute_mean" not in keys and "impute_zero" in keys


def test_each_filling_option_is_actually_scored():
    result = suggest_improvements(_with_gaps(), "target", "classification")
    for recipe in result["recipes"]:
        if recipe["key"].startswith("impute_"):
            assert recipe["score"] is not None
            assert recipe["delta"] is not None
