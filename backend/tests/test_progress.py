"""Live pipeline progress."""
from __future__ import annotations

from app.agents import progress

PLAN = ["Reading the file", "Cleaning the data", "Training models"]


def setup_function():
    progress._RUNS.clear()


def test_unknown_dataset_reports_not_running():
    snap = progress.snapshot("nope")
    assert snap["running"] is False and snap["done"] == []


def test_progress_advances_through_the_plan():
    progress.start("d1", "training", PLAN)
    assert progress.snapshot("d1")["completed"] == 0

    progress.begin_stage("d1", "Cleaning the data")
    assert progress.snapshot("d1")["current"] == "Cleaning the data"
    assert progress.snapshot("d1")["completed"] == 1


def test_repeated_calls_never_exceed_the_plan():
    """Training fires one call per model; the bar must not read '13 of 8'."""
    progress.start("d1", "training", PLAN)
    progress.begin_stage("d1", "Training models")
    for i in range(12):
        progress.end_stage("d1", "Training models", f"model {i}", 0.4)
    snap = progress.snapshot("d1")
    assert snap["completed"] <= snap["total"]
    assert len(snap["done"]) == 12  # detail is still kept


def test_finish_marks_it_complete():
    progress.start("d1", "training", PLAN)
    progress.finish("d1")
    snap = progress.snapshot("d1")
    assert snap["finished"] is True and snap["running"] is False
    assert snap["completed"] == snap["total"]


def test_failure_is_recorded():
    progress.start("d1", "training", PLAN)
    progress.finish("d1", error="ValidationError")
    assert progress.snapshot("d1")["error"] == "ValidationError"


def test_tracking_is_bounded():
    for i in range(progress.MAX_TRACKED + 10):
        progress.start(f"ds{i}", "training", PLAN)
    assert len(progress._RUNS) <= progress.MAX_TRACKED


def test_endpoint_returns_a_snapshot(client, regression_dataset_id):
    r = client.get(f"/api/datasets/{regression_dataset_id}/progress")
    assert r.status_code == 200
    assert "running" in r.json()
