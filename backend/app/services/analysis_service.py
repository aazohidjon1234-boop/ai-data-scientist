"""Run the agent's analysis pipeline and persist results."""
from __future__ import annotations

from sqlalchemy.orm import Session

from ..agents.pipeline_agent import DataScientistAgent
from ..agents import progress
from ..core.config import get_settings
from ..utils.dataframe_store import load_csv_cached
from ..models.db import Analysis
from .dataset_service import get_dataset_or_404


def run_analysis(db: Session, dataset_id: str, target_override: str | None = None) -> Analysis:
    ds = get_dataset_or_404(db, dataset_id)
    df = load_csv_cached(ds.file_path)

    agent = DataScientistAgent(df, dataset_id=ds.id)
    progress.start(ds.id, "analysis", [
        "Reading the file", "Checking missing values", "Computing statistics",
        "Measuring correlations", "Looking for outliers", "Deciding the task",
        "Drawing charts", "Writing the explanation",
    ])
    try:
        result = agent.analyze(target_override=target_override)
    except Exception as exc:
        progress.finish(ds.id, error=type(exc).__name__)
        raise
    progress.finish(ds.id)

    existing = db.query(Analysis).filter(Analysis.dataset_id == ds.id).first()
    if existing is None:
        existing = Analysis(dataset_id=ds.id)
        db.add(existing)
    existing.target_column = result["target_column"]
    existing.problem_type = result["problem_type"]
    existing.target_detection = result["target_detection"]
    existing.profile = result["profile"]
    existing.missing = result["missing"]
    existing.statistics = result["statistics"]
    existing.correlation = result["correlation"]
    existing.outliers = result["outliers"]
    existing.preview = result["preview"]
    existing.figures = result["figures"]
    existing.plan = result["plan"]
    existing.trace = result["trace"]
    existing.explanation = result["explanation"]
    ds.status = "analyzed"
    db.commit()
    db.refresh(existing)
    return existing
