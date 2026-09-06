"""Derived columns: build them correctly, and only keep the ones that earn it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.feature_engineer import build_features, compute, propose
from app.agents.improver import suggest_improvements


@pytest.fixture
def interaction() -> pd.DataFrame:
    """The target depends on a*b, which neither column shows on its own."""
    rng = np.random.default_rng(13)
    n = 400
    a = rng.uniform(-3, 3, n)
    b = rng.uniform(-3, 3, n)
    return pd.DataFrame({
        "a": a, "b": b, "noise": rng.normal(0, 1, n),
        "y": ((a * b + rng.normal(0, 0.5, n)) > 0).astype(int),
    })


def test_product_difference_and_ratio_are_computed():
    df = pd.DataFrame({"x": [2.0, 4.0, 9.0], "z": [1.0, 2.0, 3.0]})
    assert compute(df, {"kind": "product", "a": "x", "b": "z"}).tolist() == [2.0, 8.0, 27.0]
    assert compute(df, {"kind": "difference", "a": "x", "b": "z"}).tolist() == [1.0, 2.0, 6.0]
    assert compute(df, {"kind": "ratio", "a": "x", "b": "z"}).tolist() == [2.0, 2.0, 3.0]


def test_division_by_zero_becomes_missing_not_infinity():
    """inf would poison scaling and every model downstream."""
    df = pd.DataFrame({"x": [1.0, 2.0, 9.0], "z": [1.0, 0.0, 3.0]})
    result = compute(df, {"kind": "ratio", "a": "x", "b": "z"})
    assert pd.isna(result.iloc[1])          # 2/0 must not become inf
    assert np.isfinite(result.dropna()).all()


def test_constant_result_is_rejected():
    df = pd.DataFrame({"x": [2.0, 4.0, 6.0], "z": [1.0, 2.0, 3.0]})
    assert compute(df, {"kind": "ratio", "a": "x", "b": "z"}) is None  # always 2.0


def test_unknown_column_or_kind_returns_none():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    assert compute(df, {"kind": "product", "a": "x", "b": "missing"}) is None
    assert compute(df, {"kind": "magic", "a": "x", "b": "x"}) is None


def test_an_interaction_is_discovered(interaction):
    specs = propose(interaction, "y", "classification", ["a", "b", "noise"])
    assert any(s["kind"] == "product" and {s["a"], s["b"]} == {"a", "b"} for s in specs)


def test_candidates_must_beat_their_parents(interaction):
    for spec in propose(interaction, "y", "classification", ["a", "b", "noise"]):
        assert spec["relevance"] > spec["parent_relevance"]


def test_nothing_is_proposed_without_two_numeric_columns():
    df = pd.DataFrame({"only": [1.0, 2.0, 3.0] * 20, "y": [0, 1, 0] * 20})
    assert propose(df, "y", "classification", ["only"]) == []


def test_build_adds_columns_without_touching_the_original(interaction):
    specs = propose(interaction, "y", "classification", ["a", "b", "noise"])
    out, added = build_features(interaction, specs)
    assert added and all(name in out.columns for name in added)
    assert list(interaction.columns) == ["a", "b", "noise", "y"]  # unchanged
    assert len(out) == len(interaction)


def test_building_twice_does_not_duplicate(interaction):
    specs = propose(interaction, "y", "classification", ["a", "b", "noise"])
    once, added = build_features(interaction, specs)
    twice, again = build_features(once, specs)
    assert again == []
    assert len(twice.columns) == len(once.columns)


def test_recipe_is_offered_and_scored(interaction):
    result = suggest_improvements(interaction, "y", "classification")
    recipe = next((r for r in result["recipes"] if r["key"] == "engineered"), None)
    assert recipe is not None and recipe["score"] is not None
    assert recipe["changes"]["engineered"]


def test_recipe_disappears_once_applied(interaction):
    result = suggest_improvements(interaction, "y", "classification",
                                  current_engineered=["a_x_b"])
    assert all(r["key"] != "engineered" for r in result["recipes"])


def test_training_builds_and_reports_them(client, regression_dataset_id):
    r = client.post(f"/api/datasets/{regression_dataset_id}/train", json={
        "target": "price",
        "engineered": [{"kind": "product", "a": "size", "b": "rooms", "name": "size_x_rooms"}],
    })
    assert r.status_code == 200, r.text
    assert "size_x_rooms" in r.json()["run_info"]["engineered_columns"]
