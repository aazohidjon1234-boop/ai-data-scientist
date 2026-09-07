"""Prediction, cross-validated ranking, class balancing and deletion."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.tools.data_tools import class_balance, compare_models, cv_score, prepare_features
from app.tools.preprocessor import Preprocessor
from app.exceptions import ValidationError


def _frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    age = rng.integers(20, 70, n)
    city = rng.choice(["a", "b", "c"], n)
    return pd.DataFrame({"age": age, "city": city,
                         "label": ((age > 45) ^ (city == "a")).astype(int)})


# ------------------------------------------------------------- preprocessor
def test_preprocessor_reproduces_the_training_matrix():
    df = _frame()
    X, _, report = prepare_features(df, "label", "classification")
    pre = Preprocessor.from_dict(report["preprocessor"])
    again = pre.transform(df.drop(columns=["label"]))
    assert list(again.columns) == list(X.columns)
    np.testing.assert_allclose(again.to_numpy(), X.to_numpy(), rtol=1e-6, atol=1e-6)


def test_preprocessor_scales_new_rows_with_training_statistics():
    """Recomputing the mean from new rows would move the model's inputs."""
    df = _frame()
    _, _, report = prepare_features(df, "label", "classification")
    pre = Preprocessor.from_dict(report["preprocessor"])
    out = pre.transform(pd.DataFrame({"age": [40], "city": ["a"]}))
    expected = (40 - pre.scaler_mean["age"]) / pre.scaler_scale["age"]
    assert out["age"].iloc[0] == pytest.approx(expected)


def test_unseen_category_does_not_break_the_layout():
    df = _frame()
    _, _, report = prepare_features(df, "label", "classification")
    pre = Preprocessor.from_dict(report["preprocessor"])
    out = pre.transform(pd.DataFrame({"age": [30], "city": ["brand_new"]}))
    assert list(out.columns) == pre.features
    assert out.filter(like="city_").to_numpy().sum() == 0  # no known category set


def test_missing_required_column_is_reported():
    df = _frame()
    _, _, report = prepare_features(df, "label", "classification")
    pre = Preprocessor.from_dict(report["preprocessor"])
    with pytest.raises(ValidationError, match="missing"):
        pre.transform(pd.DataFrame({"age": [30]}))


def test_labels_are_decoded_back_to_the_originals():
    df = _frame().assign(label=lambda d: np.where(d.label == 1, "yes", "no"))
    _, _, report = prepare_features(df, "label", "classification")
    pre = Preprocessor.from_dict(report["preprocessor"])
    assert set(pre.decode(np.array([0, 1]))) <= {"yes", "no"}


# ------------------------------------------------------------------ predict
@pytest.fixture
def trained(client):
    csv = "age,city,label\n" + "\n".join(
        f"{20 + i % 50},{'a' if i % 3 else 'b'},{i % 2}" for i in range(160)) + "\n"
    ds = client.post("/api/datasets/upload",
                     files={"file": ("p.csv", io.BytesIO(csv.encode()), "text/csv")}).json()["id"]
    client.post(f"/api/datasets/{ds}/train", json={"target": "label"})
    return ds


def test_predict_schema_lists_the_inputs(client, trained):
    body = client.get(f"/api/datasets/{trained}/predict/schema").json()
    assert set(body["required_columns"]) == {"age", "city"}
    assert body["model"] and body["target"] == "label"


def test_predict_returns_one_label_per_row(client, trained):
    r = client.post(f"/api/datasets/{trained}/predict",
                    json={"rows": [{"age": 30, "city": "a"}, {"age": 65, "city": "b"}]})
    assert r.status_code == 200, r.text
    assert len(r.json()["predictions"]) == 2


def test_predict_reports_confidence_for_classification(client, trained):
    body = client.post(f"/api/datasets/{trained}/predict",
                       json={"rows": [{"age": 30, "city": "a"}]}).json()
    if body["confidence"] is not None:
        assert sum(body["confidence"][0].values()) == pytest.approx(1.0, abs=0.01)


def test_predict_rejects_missing_columns(client, trained):
    r = client.post(f"/api/datasets/{trained}/predict", json={"rows": [{"age": 30}]})
    assert r.status_code == 400 and "city" in r.json()["error"]["message"]


def test_predict_from_csv(client, trained):
    r = client.post(f"/api/datasets/{trained}/predict/csv",
                    files={"file": ("n.csv", io.BytesIO(b"age,city\n25,a\n60,b\n"), "text/csv")})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_predict_before_training_is_refused(client):
    csv = "a,b\n1,2\n3,4\n"
    ds = client.post("/api/datasets/upload",
                     files={"file": ("x.csv", io.BytesIO(csv.encode()), "text/csv")}).json()["id"]
    assert client.get(f"/api/datasets/{ds}/predict/schema").status_code == 400


# ------------------------------------------------------------------ ranking
def test_cross_validation_is_used_for_small_data():
    df = _frame(300)
    X, y, _ = prepare_features(df, "label", "classification")
    assert cv_score("Random Forest", "classification", X, y) is not None


def test_cross_validation_is_skipped_on_large_data():
    """The whole point is small-sample noise; big frames do not need the cost."""
    from app.tools.data_tools import CV_RANKING_MAX_ROWS

    df = _frame(CV_RANKING_MAX_ROWS + 200)
    X, y, _ = prepare_features(df, "label", "classification")
    assert cv_score("Random Forest", "classification", X, y) is None


def test_ranking_prefers_the_cv_score_when_present():
    results = [
        {"name": "lucky", "status": "ok", "metrics": {"f1": 0.90, "cv_score": 0.70}},
        {"name": "steady", "status": "ok", "metrics": {"f1": 0.85, "cv_score": 0.88}},
    ]
    out = compare_models(results, "classification")
    assert out["ranked_by"] == "cross_validation"
    assert out["best"] == "steady"          # not the one that won a single split


def test_ranking_falls_back_to_holdout_without_cv():
    results = [
        {"name": "a", "status": "ok", "metrics": {"f1": 0.90}},
        {"name": "b", "status": "ok", "metrics": {"f1": 0.85}},
    ]
    out = compare_models(results, "classification")
    assert out["ranked_by"] == "holdout" and out["best"] == "a"


def test_cv_score_survives_the_database(client, trained):
    body = client.get(f"/api/datasets/{trained}/models").json()
    assert any(m.get("cv_score") is not None for m in body["models"])


# ---------------------------------------------------------------- balancing
def test_class_balance_flags_a_lopsided_target():
    y = pd.Series([0] * 92 + [1] * 8)
    result = class_balance(y)
    assert result["imbalanced"] and result["largest_share"] == pytest.approx(0.92)


def test_class_balance_accepts_an_even_target():
    assert not class_balance(pd.Series([0, 1] * 50))["imbalanced"]


def test_balancing_reaches_training(client, trained):
    r = client.post(f"/api/datasets/{trained}/train",
                    json={"target": "label", "balance_classes": True})
    assert r.status_code == 200
    assert r.json()["run_info"]["balanced_weights"] is True


# ----------------------------------------------------------------- deletion
def test_delete_removes_the_dataset_and_its_files(client, trained, tmp_path):
    from pathlib import Path
    from app.core.config import get_settings

    models = Path(get_settings().model_dir) / trained
    assert models.exists()

    r = client.delete(f"/api/datasets/{trained}")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert "saved models" in r.json()["removed"]
    assert not models.exists()
    assert client.get(f"/api/datasets/{trained}").status_code == 404


def test_deleting_an_unknown_dataset_is_404(client):
    assert client.delete("/api/datasets/nope").status_code == 404
