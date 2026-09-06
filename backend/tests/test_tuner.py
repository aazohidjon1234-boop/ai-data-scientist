"""Hyperparameter search: it must be measured, and it must be allowed to fail."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.tuner import GRIDS, is_tunable, tune_model
from app.tools.data_tools import model_factory, prepare_features


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 320
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c,
                         "y": ((a * 2 + b - c + rng.normal(0, 0.6, n)) > 0).astype(int)})


def _xy(frame, kind="classification", target="y"):
    return prepare_features(frame, target, kind)[:2]


def test_every_grid_targets_a_real_model():
    """A typo in a grid key would silently disable tuning for that model."""
    for name in GRIDS:
        assert model_factory("classification", name) or model_factory("regression", name), name


def test_grid_parameters_are_accepted_by_the_estimator():
    for name, grid in GRIDS.items():
        factory = model_factory("classification", name) or model_factory("regression", name)
        valid = set(factory().get_params())
        unknown = set(grid) - valid
        assert not unknown, f"{name}: {unknown}"


def test_search_returns_a_measured_score(frame):
    X, y = _xy(frame)
    result = tune_model(X, y, "classification", "Random Forest", iterations=6)
    assert result is not None
    assert 0.0 <= result["cv_score"] <= 1.0
    assert set(result["params"]) <= set(GRIDS["Random Forest"])


def test_found_parameters_are_json_safe(frame):
    """numpy scalars break JSON serialisation on the way to the database."""
    import json

    X, y = _xy(frame)
    result = tune_model(X, y, "classification", "Random Forest", iterations=5)
    json.dumps(result["params"])  # must not raise


def test_found_parameters_actually_fit(frame):
    X, y = _xy(frame)
    result = tune_model(X, y, "classification", "Random Forest", iterations=5)
    model = model_factory("classification", "Random Forest")().set_params(**result["params"])
    model.fit(X, y)
    assert len(model.predict(X)) == len(y)


def test_untunable_model_returns_none(frame):
    X, y = _xy(frame)
    assert not is_tunable("Naive Bayes")
    assert tune_model(X, y, "classification", "Naive Bayes") is None


def test_regression_search_works():
    rng = np.random.default_rng(8)
    n = 260
    size = rng.normal(100, 20, n)
    df = pd.DataFrame({"size": size, "noise": rng.normal(0, 1, n),
                       "price": size * 3 + rng.normal(0, 6, n)})
    X, y = _xy(df, "regression", "price")
    result = tune_model(X, y, "regression", "Ridge", iterations=5)
    assert result and result["metric"] == "R²"


def test_training_keeps_defaults_when_tuning_is_worse(client, regression_dataset_id):
    """The search optimises CV folds; the held-out split decides."""
    r = client.post(f"/api/datasets/{regression_dataset_id}/train",
                    json={"target": "price", "tune": True})
    assert r.status_code == 200, r.text
    report = r.json()["run_info"].get("tuning")
    if report is None:
        pytest.skip("winning model has no grid")
    assert "applied" in report
    if report["applied"]:
        assert report["after"] > report["before"]
    else:
        assert report["after"] is None or report["after"] <= report["before"]
