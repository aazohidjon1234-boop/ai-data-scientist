"""Report generation: markdown assembled from real results, saved to disk."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..exceptions import ValidationError
from ..models.db import Report
from ..tools.report_builder import generate_report
from .dataset_service import get_dataset_or_404
from .ml_service import training_from_db


def _analysis_dict(ds, analysis) -> dict:
    return {
        "profile": analysis.profile or {},
        "missing": analysis.missing or {},
        "statistics": analysis.statistics or {},
        "correlation": analysis.correlation,
        "outliers": analysis.outliers or {},
        "preview": analysis.preview or {},
        "target_detection": analysis.target_detection or {},
        "problem_type": analysis.problem_type if analysis else None,
        "target_column": analysis.target_column if analysis else None,
        "explanation": analysis.explanation if analysis else "",
    }


def run_report(db: Session, dataset_id: str) -> Report:
    ds = get_dataset_or_404(db, dataset_id)
    analysis = ds.analysis
    if analysis is None:
        raise ValidationError("Analyze the dataset before generating a report.")

    training = training_from_db(db, ds)
    a_dict = _analysis_dict(ds, analysis)

    content = generate_report(ds.name, a_dict, training,
                              (training or {}).get("clean_report"))

    settings = get_settings()
    Path(settings.report_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(settings.report_dir) / f"report_{ds.id}_{stamp}.md"
    path.write_text(content, encoding="utf-8")

    report = Report(dataset_id=ds.id, format="markdown", content_path=str(path))
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def report_content(report: Report) -> str:
    p = Path(report.content_path)
    if not p.exists():
        return "_Report file is missing from disk._"
    return p.read_text(encoding="utf-8")
