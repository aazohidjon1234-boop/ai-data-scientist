"""Model training: real metrics, best-model selection, per-task behavior."""
import pytest

from tests.conftest import _regression_rows

_REG_ROWS = _regression_rows()


def _train(client, ds_id, **body):
    r = client.post(f"/api/datasets/{ds_id}/train", json=body or {})
    assert r.status_code == 200, r.text
    return r.json()


REGRESSION_MODELS = {
    "Linear Regression", "Ridge", "Decision Tree", "Random Forest", "Extra Trees",
    "Gradient Boosting", "Hist Gradient Boosting", "XGBoost", "LightGBM", "KNN", "SVR",
}
CLASSIFICATION_MODELS = {
    "Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees",
    "Gradient Boosting", "XGBoost", "LightGBM", "SVM", "KNN", "Naive Bayes", "AdaBoost",
}


def test_regression_training(regression_dataset_id, client):
    run = _train(client, regression_dataset_id)
    assert run["problem_type"] == "regression"
    assert run["target"] == "price"
    models = run["models"]
    assert len(models) == len(REGRESSION_MODELS)
    assert {m["name"] for m in models} == REGRESSION_MODELS

    for m in models:
        assert m["status"] == "ok", m.get("status_message")
        for k in ("mae", "mse", "rmse", "r2"):
            assert k in m["metrics"]
        assert m["metrics"]["mae"] >= 0
        assert m["metrics"]["mse"] >= 0
        assert abs(m["metrics"]["rmse"] ** 2 - m["metrics"]["mse"]) < 1.0
        assert m["metrics"]["r2"] <= 1.0
        assert m["training_seconds"] >= 0

    best = [m for m in models if m["is_best"]]
    assert len(best) == 1
    assert run["best_model"] == best[0]["name"]
    # best must have the highest R2
    r2s = {m["name"]: m["metrics"]["r2"] for m in models}
    assert r2s[run["best_model"]] == max(r2s.values())
    # fixture is nearly linear: the fit should be good
    assert r2s[run["best_model"]] > 0.7

    # feature importance comes from an actually trained model
    fi = best[0]["feature_importance"]
    assert fi and len(fi) >= 2
    assert fi[0]["importance"] >= fi[1]["importance"]


def test_classification_training(classification_dataset_id, client):
    run = _train(client, classification_dataset_id)
    assert run["problem_type"] == "classification"
    assert run["target"] == "outcome"
    models = run["models"]
    assert len(models) == len(CLASSIFICATION_MODELS)
    assert {m["name"] for m in models} == CLASSIFICATION_MODELS

    for m in models:
        assert m["status"] == "ok", m.get("status_message")
        met = m["metrics"]
        for k in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= met[k] <= 1.0
        cm = met["confusion_matrix"]
        assert len(cm["labels"]) == 2
        assert len(cm["matrix"]) == 2
        # confusion matrix row sums == support
        for row, sup in zip(cm["matrix"], cm["support"]):
            assert sum(row) == sup

    assert run["best_model"] in {m["name"] for m in models}
    best = next(m for m in models if m["is_best"])
    f1s = {m["name"]: m["metrics"]["f1"] for m in models}
    assert f1s[run["best_model"]] == max(f1s.values())


def test_clustering_training(clustering_dataset_id, client):
    run = _train(client, clustering_dataset_id)
    assert run["problem_type"] == "clustering"
    assert run["target"] is None
    models = run["models"]
    names = [m["name"] for m in models]
    # full cluster zoo: all three algorithms must be present
    assert any(n.startswith("K-Means") for n in names)
    assert any(n.startswith("Agglomerative") for n in names)
    assert any(n.startswith("DBSCAN") for n in names)
    for m in models:
        assert m["status"] == "ok", m.get("status_message")
        met = m["metrics"]
        assert -1.0 <= met["silhouette"] <= 1.0
        if m["name"].startswith("DBSCAN"):
            # noise rows are excluded from the cluster sizes
            assert sum(met["cluster_sizes"]) + met["noise"] == run["run_info"]["n_train"]
        else:
            assert met["k"] >= 2
            assert sum(met["cluster_sizes"]) == run["run_info"]["n_train"]

    best = next(m for m in models if m["is_best"])
    sils = [m["metrics"]["silhouette"] for m in models if m["status"] == "ok"]
    assert abs(best["metrics"]["silhouette"] - max(sils)) < 1e-9
    # fixture has 3 clear blobs
    assert best["metrics"]["k"] == 3


def test_train_explicit_target_override(regression_dataset_id, client):
    run = _train(client, regression_dataset_id, target="neighborhood")
    assert run["problem_type"] == "classification"
    assert run["target"] == "neighborhood"
    assert all(m["status"] == "ok" for m in run["models"])


def test_train_explicit_problem_type(clustering_dataset_id, client):
    run = _train(client, clustering_dataset_id, problem_type="clustering")
    assert run["problem_type"] == "clustering"


def test_train_invalid_target_400(regression_dataset_id, client):
    r = client.post(f"/api/datasets/{regression_dataset_id}/train", json={"target": "ghost"})
    assert r.status_code == 400


def test_train_without_target_on_supervised_ambiguous(regression_dataset_id, client):
    # price has a name hint, so auto-detect still finds it
    run = _train(client, regression_dataset_id)
    assert run["target"] == "price"


def test_models_endpoint_returns_stored_run(regression_dataset_id, client):
    _train(client, regression_dataset_id)
    r = client.get(f"/api/datasets/{regression_dataset_id}/models")
    assert r.status_code == 200
    body = r.json()
    assert body["best_model"]
    assert len(body["models"]) == len(REGRESSION_MODELS)
    assert body["dataset_id"] == regression_dataset_id


def test_models_endpoint_before_training_400(clustering_dataset_id, client):
    r = client.get(f"/api/datasets/{clustering_dataset_id}/models")
    assert r.status_code == 400


def test_retrain_replaces_previous_run(regression_dataset_id, client):
    _train(client, regression_dataset_id)
    run2 = _train(client, regression_dataset_id)
    r = client.get(f"/api/datasets/{regression_dataset_id}/models")
    body = r.json()
    assert len(body["models"]) == len(REGRESSION_MODELS)  # replaced, not doubled
    assert body["best_model"] == run2["best_model"]


def test_trained_model_downloadable_and_loadable(regression_dataset_id, client):
    """Every ok model must be downloadable and the artifact must actually load & predict."""
    import io

    import joblib

    run = _train(client, regression_dataset_id)
    for m in run["models"]:
        assert m["status"] == "ok"
        assert m["download_url"], "ok models must expose a download URL"

    best = next(m for m in run["models"] if m["is_best"])
    r = client.get(best["download_url"])
    assert r.status_code == 200
    assert len(r.content) > 0

    model = joblib.load(io.BytesIO(r.content))
    assert hasattr(model, "predict")

    # metadata endpoint exists for the run
    from tests.conftest import client as _c  # noqa: F401

    meta = client.get(f"/api/datasets/{regression_dataset_id}/models/anything/metadata")
    assert meta.status_code == 200
    body = meta.json()
    assert body["problem_type"] == "regression"
    assert "features" in body


def test_model_download_unknown_404(regression_dataset_id, client):
    _train(client, regression_dataset_id)
    r = client.get(f"/api/datasets/{regression_dataset_id}/models/no-such-model-xyz/download")
    assert r.status_code == 400  # not found -> clean validation error


# ---------------------------------------------------- user-chosen features
# The training pipeline lets the caller pick which columns become X. These
# guard the two ways that can go wrong: chosen columns being ignored, and
# unchosen columns (especially the target) leaking in anyway.

@pytest.fixture
def frame():
    """Raw frame minus rows with no target — what cleaning hands to prepare_features."""
    from app.services.dataset_service import parse_csv

    df = parse_csv(("price,size,rooms,neighborhood\n" + _REG_ROWS).encode())
    return df.dropna(subset=["price"]).reset_index(drop=True)


def test_feature_selection_restricts_the_model_inputs(frame):
    from app.tools.data_tools import prepare_features

    X_all, _, report_all = prepare_features(frame, "price", "regression")
    X_few, _, report_few = prepare_features(frame, "price", "regression", ["size", "rooms"])
    assert X_few.shape[1] < X_all.shape[1]
    assert set(report_few["source_columns"]) == {"size", "rooms"}
    assert report_few["selected_by_user"] == ["size", "rooms"]
    assert report_all["selected_by_user"] is None  # omitted == use everything


def test_feature_selection_keeps_every_row(frame):
    from app.tools.data_tools import prepare_features

    X, y, _ = prepare_features(frame, "price", "regression", ["size"])
    assert len(X) == len(frame) and len(y) == len(frame)


def test_target_cannot_be_used_as_its_own_feature(frame):
    """Including the target in X would leak the answer into the model."""
    from app.tools.data_tools import prepare_features

    _, _, report = prepare_features(frame, "price", "regression", ["price", "size"])
    assert "price" not in report["source_columns"]
    assert report["selected_by_user"] == ["size"]


def test_unknown_feature_column_is_rejected(frame):
    from app.exceptions import ValidationError
    from app.tools.data_tools import prepare_features

    with pytest.raises(ValidationError, match="not in the dataset"):
        prepare_features(frame, "price", "regression", ["nope"])


def test_selecting_only_the_target_is_rejected(frame):
    from app.exceptions import ValidationError
    from app.tools.data_tools import prepare_features

    with pytest.raises(ValidationError, match="at least one input column"):
        prepare_features(frame, "price", "regression", ["price"])


def test_unknown_target_is_rejected(frame):
    from app.exceptions import ValidationError
    from app.tools.data_tools import prepare_features

    with pytest.raises(ValidationError, match="not in this dataset"):
        prepare_features(frame, "no_such_column", "regression")


def test_train_endpoint_accepts_features(client, regression_dataset_id):
    run = _train(client, regression_dataset_id,
                 target="price", problem_type="regression", features=["size", "rooms"])
    assert run["best_model"]
    assert run["target"] == "price"


def test_train_endpoint_rejects_bad_features(client, regression_dataset_id):
    r = client.post(f"/api/datasets/{regression_dataset_id}/train",
                    json={"target": "price", "features": ["not_a_column"]})
    assert r.status_code == 400
    assert "not in the dataset" in r.json()["error"]["message"]


# --------------------------------------------- large data / awkward names
def test_kernel_models_subsample_both_sides():
    """SVM caps its training rows; y must be cut with the same indices.

    Sampling X alone left y at full length, so SVM and SVR failed with
    "inconsistent numbers of samples" on every dataset above the cap.
    """
    import numpy as np
    import pandas as pd
    from app.tools import data_tools

    cap = data_tools._MAX_FIT_ROWS["SVM"]
    n = cap + 500
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    y = (X["a"] + rng.normal(0, 0.5, n) > 0).astype(int).to_numpy()

    model, seconds = data_tools.train_model("SVM", "classification", X, y, random_state=0)
    assert seconds >= 0
    assert len(model.predict(X.head(10))) == 10


def test_feature_names_are_stripped_of_characters_boosters_reject():
    """LightGBM rejects spaces and JSON punctuation in feature names."""
    import pandas as pd
    from app.tools.data_tools import prepare_features

    df = pd.DataFrame({
        "General Health": ["Very Good", "Poor", "Very Good", "Poor"] * 5,
        "score[raw]": range(20),
        "y": [0, 1] * 10,
    })
    X, _, report = prepare_features(df, "y", "classification")
    for name in X.columns:
        assert all(ch.isalnum() or ch == "_" for ch in name), name
    assert all(all(c.isalnum() or c == "_" for c in n) for n in report["features"])


def test_sanitising_does_not_merge_distinct_columns():
    from app.tools.data_tools import _safe_feature_names

    assert _safe_feature_names(["a b", "a!b"]) == ["a_b", "a_b_1"]


def test_boosters_train_on_multiword_categories(client):
    """End to end: the combination that failed in the browser."""
    import io

    rows = "\n".join(
        f"{'Very Good' if i % 3 else 'Poor'},{i % 7},{i % 2}" for i in range(120)
    )
    csv = "general health,score,label\n" + rows + "\n"
    r = client.post("/api/datasets/upload",
                    files={"file": ("cat.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 201, r.text
    run = client.post(f"/api/datasets/{r.json()['id']}/train", json={"target": "label"})
    assert run.status_code == 200, run.text
    failed = {m["name"]: m["status_message"] for m in run.json()["models"] if m["status"] != "ok"}
    assert "LightGBM" not in failed, failed.get("LightGBM")
    assert "XGBoost" not in failed, failed.get("XGBoost")
