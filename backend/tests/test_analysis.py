"""Dataset analysis: profile, missing values, statistics, correlations, figures."""
def _analyze(client, ds_id, **body):
    r = client.post(f"/api/datasets/{ds_id}/analyze", json=body or {})
    assert r.status_code == 200, r.text
    return r.json()


def test_analyze_basic_profile(regression_dataset_id, client):
    a = _analyze(client, regression_dataset_id)
    p = a["profile"]
    # 60 rows + 3 duplicates
    assert p["rows"] == 63
    assert p["columns"] == 4
    assert p["duplicate_rows"] == 3
    assert "price" in p["numeric_columns"]
    assert "neighborhood" in p["categorical_columns"]


def test_missing_value_detection(regression_dataset_id, client):
    a = _analyze(client, regression_dataset_id)
    missing = a["missing"]
    # we injected one missing price and one missing size
    per = {c["column"]: c for c in missing["per_column"]}
    assert per["price"]["missing"] == 1
    assert per["size"]["missing"] == 1
    assert missing["total_missing"] == 2


def test_statistics_present(regression_dataset_id, client):
    a = _analyze(client, regression_dataset_id)
    stats = a["statistics"]
    assert stats["price"]["kind"] == "numeric"
    for k in ("mean", "median", "std", "min", "q1", "q3", "max"):
        assert k in stats["price"]
    assert stats["neighborhood"]["kind"] == "categorical"
    assert stats["neighborhood"]["unique"] == 3


def test_correlation_matrix(regression_dataset_id, client):
    a = _analyze(client, regression_dataset_id)
    corr = a["correlation"]
    assert corr is not None
    assert len(corr["columns"]) == 3  # price, size, rooms
    assert len(corr["matrix"]) == 3
    # price and size are strongly correlated in the fixture
    top = corr["top_pairs"][0]
    assert {top["a"], top["b"]} == {"price", "size"}
    assert abs(top["corr"]) > 0.9


def test_target_detection_regression(regression_dataset_id, client):
    a = _analyze(client, regression_dataset_id)
    assert a["problem_type"] == "regression"
    assert a["target_column"] == "price"
    assert a["target_detection"]["auto_detected"] is True


def test_target_override_to_classification(regression_dataset_id, client):
    a = _analyze(client, regression_dataset_id, target="neighborhood")
    assert a["problem_type"] == "classification"
    assert a["target_column"] == "neighborhood"


def test_target_override_invalid_400(regression_dataset_id, client):
    r = client.post(f"/api/datasets/{regression_dataset_id}/analyze", json={"target": "nope"})
    assert r.status_code == 400


def test_visualizations_generated(regression_dataset_id, client):
    _analyze(client, regression_dataset_id)
    r = client.get(f"/api/datasets/{regression_dataset_id}/visualizations")
    assert r.status_code == 200
    figures = r.json()["figures"]
    kinds = {f["kind"] for f in figures}
    assert "histogram" in kinds
    assert "heatmap" in kinds
    assert "bar" in kinds
    for f in figures:
        assert f["data"]["data"], "figure must contain plotly traces"
        assert f["data"]["layout"], "figure must contain a layout"


def test_analysis_trace_and_plan(regression_dataset_id, client):
    a = _analyze(client, regression_dataset_id)
    tools = [s["tool"] for s in a["trace"]]
    for expected in (
        "analyze_dataset",
        "detect_missing_values",
        "generate_statistics",
        "generate_correlation_matrix",
        "detect_outliers",
        "detect_problem_type",
        "create_visualization",
    ):
        assert expected in tools
    assert all(s["status"] in ("ok", "skipped") for s in a["trace"])
    assert len(a["plan"]) >= 6
    assert len(a["explanation"]) > 200


def test_preview_shape(regression_dataset_id, client):
    a = _analyze(client, regression_dataset_id)
    pv = a["preview"]
    assert len(pv["columns"]) == 4
    assert len(pv["rows"]) <= 100
    assert len(pv["rows"][0]) == 4


def test_explanation_uses_real_numbers(regression_dataset_id, client):
    """The explanation must cite numbers that exist in the computed profile."""
    a = _analyze(client, regression_dataset_id)
    assert str(a["profile"]["rows"]) in a["explanation"]
    assert str(a["profile"]["columns"]) in a["explanation"]
