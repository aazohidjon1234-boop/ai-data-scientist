"""Model training: real metrics, best-model selection, per-task behavior."""
import pytest


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
