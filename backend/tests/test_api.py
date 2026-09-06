"""End-to-end API flow: upload -> analyze -> train -> models -> viz -> chat -> report."""
from tests.conftest import upload_csv


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_full_happy_path(regression_dataset_id, client):
    ds_id = regression_dataset_id

    # dataset detail before analysis
    r = client.get(f"/api/datasets/{ds_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["analysis"] is None
    assert detail["models"] is None

    # analyze
    a = client.post(f"/api/datasets/{ds_id}/analyze", json={}).json()
    assert a["problem_type"] == "regression"

    # detail now includes analysis
    detail = client.get(f"/api/datasets/{ds_id}").json()
    assert detail["analysis"]["target_column"] == "price"

    # train
    t = client.post(f"/api/datasets/{ds_id}/train", json={}).json()
    assert t["best_model"]
    assert len(t["trace"]) >= 5

    # models
    m = client.get(f"/api/datasets/{ds_id}/models").json()
    assert m["models"]

    # visualizations
    v = client.get(f"/api/datasets/{ds_id}/visualizations").json()
    assert v["figures"]

    # chat
    c = client.post(f"/api/datasets/{ds_id}/chat", json={"message": "What does R2 mean?"})
    assert c.status_code == 200
    assert "R²" in c.json()["reply"] or "R-squared" in c.json()["reply"]

    # report
    rep = client.post(f"/api/datasets/{ds_id}/report")
    assert rep.status_code == 200
    body = rep.json()
    assert "## 1. Dataset Overview" in body["content"]
    assert "Model Comparison" in body["content"]

    # download
    dl = client.get(body["download_url"])
    assert dl.status_code == 200
    assert "Dataset Analysis Report" in dl.text

    # list shows training state
    lst = client.get("/api/datasets").json()
    ds = next(d for d in lst if d["id"] == ds_id)
    assert ds["has_training"] is True
    assert ds["best_model"] == t["best_model"]


def test_report_requires_analysis(clustering_dataset_id, client):
    r = client.post(f"/api/datasets/{clustering_dataset_id}/report")
    assert r.status_code == 400


def test_visualizations_require_analysis(regression_dataset_id, client):
    # (analysis may already have run in another test for a different dataset id —
    # use a fresh dataset to be safe)
    r = upload_csv(client, "x,y\n1,2\n3,4\n5,6\n", "fresh.csv")
    ds_id = r.json()["id"]
    v = client.get(f"/api/datasets/{ds_id}/visualizations")
    assert v.status_code == 400


def test_chat_before_analysis(client):
    r = upload_csv(client, "x,y\n1,2\n3,4\n", "chatme.csv")
    ds_id = r.json()["id"]
    c = client.post(f"/api/datasets/{ds_id}/chat", json={"message": "Summarize this"})
    assert c.status_code == 200
    assert "not been analyzed" in c.json()["reply"]


def test_chat_questions(regression_dataset_id, client):
    client.post(f"/api/datasets/{regression_dataset_id}/analyze", json={})
    client.post(f"/api/datasets/{regression_dataset_id}/train", json={})

    c = client.post(
        f"/api/datasets/{regression_dataset_id}/chat", json={"message": "Which features matter most?"}
    ).json()
    assert "square" not in c["reply"].lower() or "feature" in c["reply"].lower()
    assert len(c["reply"]) > 30

    c2 = client.post(
        f"/api/datasets/{regression_dataset_id}/chat", json={"message": "What missing values did you fix?"}
    ).json()
    assert "price" in c2["reply"] or "size" in c2["reply"]

    c3 = client.post(
        f"/api/datasets/{regression_dataset_id}/chat", json={"message": "Why is that the best model?"}
    ).json()
    assert "best" in c3["reply"].lower()


def test_sample_listing(client):
    r = client.get("/api/datasets/samples")
    assert r.status_code == 200
    # sample dir in tests may be empty (env SAMPLE_DIR points to temp dir)
    assert isinstance(r.json(), list)


def test_load_unknown_sample_404(client):
    r = client.post("/api/datasets/load-sample", json={"name": "nope"})
    assert r.status_code == 404
