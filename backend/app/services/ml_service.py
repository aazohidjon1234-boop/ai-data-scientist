"""Run the agent's training pipeline and persist model results."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..agents.pipeline_agent import DataScientistAgent, K_RANGE
from ..agents import progress
from ..core.config import get_settings
from ..tools import data_tools
from ..utils.dataframe_store import load_csv_cached
from ..models.db import ModelResult
from .dataset_service import clear_ml_state, get_dataset_or_404


def run_training(
    db: Session,
    dataset_id: str,
    target: str | None = None,
    problem_type: str | None = None,
    k_range: tuple[int, int] | None = None,
    features: list[str] | None = None,
    drop_outliers: bool = False,
    tune: bool = False,
    impute_numeric: str = "median",
    impute_categorical: str = "mode",
) -> dict[str, Any]:
    ds = get_dataset_or_404(db, dataset_id)
    df = load_csv_cached(ds.file_path)

    save_dir = Path(get_settings().model_dir) / ds.id
    agent = DataScientistAgent(df, model_dir=save_dir, dataset_id=ds.id)
    # Must mirror the tools train() actually calls, or a stage that never runs
    # sits unticked for the whole run and reads as if something stalled.
    stages = ["Reading the file", "Cleaning the data"]
    if drop_outliers:
        stages.append("Looking for outliers")
    stages += [
        "Deciding the task", "Preparing features", "Training models",
        "Evaluating on held-out data", "Comparing models", "Writing the explanation",
    ]
    progress.start(ds.id, "training", stages)
    try:
        result = agent.train(target=target, problem_type=problem_type, k_range=k_range,
                             features=features, drop_outliers=drop_outliers,
                             tune=tune, impute_numeric=impute_numeric,
                             impute_categorical=impute_categorical)
    except Exception as exc:
        progress.finish(ds.id, error=type(exc).__name__)
        raise
    progress.finish(ds.id)

    run_id = uuid.uuid4().hex
    clear_ml_state(db, ds)

    for m in result["models"]:
        db.add(
            ModelResult(
                dataset_id=ds.id,
                run_id=run_id,
                name=m["name"],
                model_type=m["model_type"],
                status=m.get("status", "ok"),
                status_message=m.get("status_message", ""),
                primary_metric=m.get("metrics", {}).get(m.get("metrics", {}).get("primary_metric", "r2")) if m.get("metrics") else None,
                metrics=m.get("metrics", {}),
                feature_importance=m.get("feature_importance"),
                training_seconds=float(m.get("training_seconds", 0.0)),
                is_best=bool(m.get("is_best", False)),
            )
        )

    ds.ml_run = {
        "run_id": run_id,
        "problem_type": result["problem_type"],
        "target": result["target"],
        "run_info": result["run_info"],
        "best_model": result["best_model"],
        "comparison": result["comparison"],
        "clean_report": result["clean_report"],
        "prep_report": result["prep_report"],
        "explanation": result["explanation"],
        "explanation_source": result.get("explanation_source", "local-engine"),
        "trace": result["trace"],
        "total_seconds": result["total_seconds"],
    }
    ds.status = "trained"
    db.commit()
    return result


def training_from_db(db: Session, ds) -> dict[str, Any] | None:
    """Rebuild the training payload from stored ModelResult rows + dataset.ml_run."""
    run = ds.ml_run
    if not run:
        return None
    rows = (
        db.execute(
            select(ModelResult)
            .where(ModelResult.dataset_id == ds.id, ModelResult.run_id == run["run_id"])
            .order_by(ModelResult.created_at)
        )
        .scalars()
        .all()
    )
    models = [
        {
            "name": r.name,
            "model_type": r.model_type,
            "status": r.status,
            "status_message": r.status_message,
            "primary_metric": r.primary_metric,
            "metrics": r.metrics or {},
            "feature_importance": r.feature_importance,
            "training_seconds": r.training_seconds,
            "is_best": r.is_best,
        }
        for r in rows
    ]
    # recompute ranks in case rows arrived unordered
    comparison = data_tools.compare_models(models, run["problem_type"])
    for m in models:
        m["rank"] = next((c["rank"] for c in comparison["ranked"] if c["name"] == m["name"]), None)
    return {
        "problem_type": run["problem_type"],
        "target": run["target"],
        "run_info": run.get("run_info", {}),
        "models": models,
        "best_model": run.get("best_model"),
        "comparison": comparison,
        "clean_report": run.get("clean_report"),
        "prep_report": run.get("prep_report"),
        "explanation": run.get("explanation", ""),
        "explanation_source": run.get("explanation_source", "local-engine"),
        "trace": run.get("trace", []),
        "total_seconds": run.get("total_seconds", 0),
        "run_id": run["run_id"],
    }
