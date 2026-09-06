"""Problem type detection unit tests."""
import numpy as np
import pandas as pd

from app.tools import data_tools


def _det(df, target=None):
    return data_tools.detect_problem_type(df, target=target)


def test_numeric_target_with_hint_is_regression():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 50),
            "sale_price": rng.uniform(100, 1000, 50),
        }
    )
    d = _det(df)
    assert d["problem_type"] == "regression"
    assert d["target"] == "sale_price"


def test_binary_categorical_target_is_classification():
    df = pd.DataFrame(
        {
            "x": np.arange(40.0),
            "churned": (["Yes", "No"] * 20),
        }
    )
    d = _det(df)
    assert d["problem_type"] == "classification"
    assert d["target"] == "churned"


def test_boolean_target_is_classification():
    df = pd.DataFrame({"a": np.arange(30.0), "flag": rng_bool(30)})
    d = _det(df)
    assert d["problem_type"] == "classification"
    assert d["target"] == "flag"


def test_no_clear_target_defaults_to_clustering():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "alpha": rng.normal(0, 1, 60),
            "beta": rng.normal(5, 2, 60),
            "gamma": rng.choice(["p", "q", "r"], 60),
        }
    )
    d = _det(df)
    assert d["problem_type"] == "clustering"
    assert d["target"] is None


def test_explicit_target_wins():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "alpha": rng.normal(0, 1, 60),
            "beta": rng.normal(5, 2, 60),
        }
    )
    d = _det(df, target="alpha")
    assert d["target"] == "alpha"
    assert d["problem_type"] == "regression"


def test_explicit_missing_target_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    import pytest

    with pytest.raises(Exception):
        _det(df, target="missing_col")


def rng_bool(n):
    rng = np.random.default_rng(5)
    return rng.random(n) > 0.5
