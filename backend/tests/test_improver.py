"""Improvement recipes: measured, never assumed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.improver import outlier_mask, suggest_improvements
from app.exceptions import ValidationError


@pytest.fixture
def clinical() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 260
    age = rng.normal(55, 9, n)
    pressure = rng.normal(130, 15, n)
    noise = rng.normal(0, 1, n)
    risk = (age - 55) * 0.12 + (pressure - 130) * 0.05 + rng.normal(0, 0.6, n)
    df = pd.DataFrame({
        "age": age.round(1), "pressure": pressure.round(1),
        "noise": noise.round(3), "target": (risk > 0).astype(int),
    })
    df.loc[:5, "pressure"] = 400.0     # unmistakable outliers
    return df


def test_outlier_mask_flags_the_extremes(clinical):
    mask = outlier_mask(clinical, ["age", "pressure", "noise"])
    assert mask.iloc[:6].all()
    assert mask.sum() < len(clinical) * 0.5


def test_outlier_mask_ignores_discrete_codes():
    df = pd.DataFrame({"code": [0, 1, 2] * 40})
    assert not outlier_mask(df, ["code"]).any()


def test_every_recipe_is_scored_against_the_baseline(clinical):
    result = suggest_improvements(clinical, "target", "classification")
    keys = {r["key"] for r in result["recipes"]}
    assert "baseline" in keys
    baseline = next(r for r in result["recipes"] if r["key"] == "baseline")
    assert baseline["score"] is not None
    for recipe in result["recipes"]:
        if recipe["score"] is not None:
            assert recipe["delta"] == pytest.approx(recipe["score"] - baseline["score"], abs=1e-4)


def test_outlier_recipe_uses_fewer_rows(clinical):
    result = suggest_improvements(clinical, "target", "classification")
    recipe = next((r for r in result["recipes"] if r["key"] == "outliers"), None)
    if recipe is None:
        pytest.skip("no outlier recipe was offered for this frame")
    baseline = next(r for r in result["recipes"] if r["key"] == "baseline")
    assert recipe["rows_used"] < baseline["rows_used"]
    assert recipe["changes"]["drop_outliers"] is True


def test_recommendation_requires_a_real_gain(clinical):
    """A recipe is only recommended when it measurably beats the baseline."""
    result = suggest_improvements(clinical, "target", "classification")
    if result["recommended"] == "baseline":
        return
    winner = next(r for r in result["recipes"] if r["key"] == result["recommended"])
    assert winner["delta"] is not None and winner["delta"] >= 0.005


def test_summary_reports_a_loss_honestly(clinical):
    result = suggest_improvements(clinical, "target", "classification")
    assert any(w in result["summary"] for w in ("Best option", "Nothing beat", "could not be scored"))


def test_regression_is_supported():
    rng = np.random.default_rng(12)
    n = 240
    size = rng.normal(100, 20, n)
    df = pd.DataFrame({"size": size, "junk": rng.normal(0, 1, n),
                       "price": size * 4 + rng.normal(0, 8, n)})
    result = suggest_improvements(df, "price", "regression")
    assert result["metric"] == "R²"
    assert result["baseline_score"] is not None


def test_clustering_is_rejected(clinical):
    with pytest.raises(ValidationError, match="clustering"):
        suggest_improvements(clinical, "target", "clustering")


def test_unknown_target_is_rejected(clinical):
    with pytest.raises(ValidationError, match="not in this dataset"):
        suggest_improvements(clinical, "nope", "classification")


def test_training_can_actually_drop_outliers(client, regression_dataset_id):
    r = client.post(f"/api/datasets/{regression_dataset_id}/train",
                    json={"target": "price", "drop_outliers": True})
    assert r.status_code == 200, r.text
    assert r.json()["best_model"]
