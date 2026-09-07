"""Dataset API: upload, samples, analysis, training, models, viz, chat, reports."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.database import get_db
from ..core.security import model_slug, safe_path
from ..exceptions import AppError, DatasetNotFoundError, ReportNotFoundError, ValidationError
from ..models.db import Analysis, Dataset, ModelResult, Report
from ..schemas.api import (
    AnalyzeRequest,
    DashboardRequest,
    ImproveRequest,
    AnalysisOut,
    AskOut,
    AskRequest,
    ChatOut,
    ChatRequest,
    DatasetDetail,
    DatasetOut,
    InsightsOut,
    LoadSampleRequest,
    ModelResultOut,
    ModelRunOut,
    PivotRequest,
    PredictRequest,
    ReportOut,
    SampleInfo,
    SegmentRequest,
    SuggestFeaturesRequest,
    TrainRequest,
    TrendRequest,
)
from ..tools.query_tools import Filter, QuerySpec
from ..utils.dataframe_store import load_csv_cached
from ..utils.jsonutils import to_jsonable
from ..services import (
    analysis_service,
    analyst_service,
    chat_service,
    dataset_service,
    ml_service,
    predict_service,
    report_service,
)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------- helpers
def _dataset_out(ds: Dataset) -> dict[str, Any]:
    run = ds.ml_run or {}
    return {
        "id": ds.id,
        "name": ds.name,
        "original_filename": ds.original_filename,
        "rows": ds.rows,
        "columns": ds.columns,
        "column_names": ds.column_names or [],
        "source": ds.source,
        "status": ds.status,
        "created_at": ds.created_at,
        "has_analysis": ds.analysis is not None,
        "has_training": bool(ds.ml_run),
        "best_model": run.get("best_model"),
        "problem_type": run.get("problem_type") or (ds.analysis.problem_type if ds.analysis else None),
        "target": run.get("target") or (ds.analysis.target_column if ds.analysis else None),
    }


def _analysis_out(a: Analysis) -> dict[str, Any]:
    return to_jsonable(
        {
            "id": a.id,
            "dataset_id": a.dataset_id,
            "target_column": a.target_column,
            "problem_type": a.problem_type,
            "target_candidates": (a.target_detection or {}).get("candidates", []),
            "target_detection": a.target_detection or {},
            "profile": a.profile or {},
            "missing": a.missing or {},
            "statistics": a.statistics or {},
            "correlation": a.correlation,
            "outliers": a.outliers or {},
            "preview": a.preview or {},
            "figures": a.figures or [],
            "plan": a.plan or [],
            "trace": a.trace or [],
            "explanation": a.explanation or "",
            "created_at": a.created_at,
        }
    )


def _training_out(db: Session, ds: Dataset) -> dict[str, Any] | None:
    t = ml_service.training_from_db(db, ds)
    if t is None:
        return None
    run = ds.ml_run or {}
    prep = run.get("prep_report") or {}
    models = []
    for m in t["models"]:
        m = dict(m)
        m["cv_score"] = (m.get("metrics") or {}).get("cv_score")
        m["cv_std"] = (m.get("metrics") or {}).get("cv_std")
        m["download_url"] = (
            f"/api/datasets/{ds.id}/models/{model_slug(m['name'])}/download"
            if m.get("status") == "ok"
            else None
        )
        models.append(m)
    return to_jsonable(
        {
            "dataset_id": ds.id,
            "run_id": run.get("run_id"),
            "problem_type": t["problem_type"],
            "target": t["target"],
            "run_info": t.get("run_info", {}),
            "ranked_by": (t.get("comparison") or {}).get("ranked_by"),
            # What this run actually learned from, so the UI can restore the
            # selection instead of silently resetting to every column.
            "features_used": prep.get("selected_by_user"),
            "source_columns": prep.get("source_columns", []),
            "drop_outliers": bool((run.get("run_info") or {}).get("outliers_dropped")),
            "models": models,
            "best_model": t.get("best_model"),
            "explanation": t.get("explanation", ""),
            "trace": t.get("trace", []),
            "created_at": ds.created_at,
        }
    )


def _get_ds(db: Session, dataset_id: str) -> Dataset:
    return dataset_service.get_dataset_or_404(db, dataset_id)


# ---------------------------------------------------------------- endpoints
@router.post("/datasets/upload", response_model=DatasetOut, status_code=201)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    from ..core.security import read_validated_csv

    content, safe_name = await read_validated_csv(file)
    df = dataset_service.parse_csv(content)

    dataset_service.ensure_dirs()
    import uuid

    ds_id = uuid.uuid4().hex
    dest_dir = safe_path(get_settings().upload_subdir, ds_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_name
    dest.write_bytes(content)
    ds = dataset_service.register_dataset(
        db, df,
        name=safe_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title() or "Dataset",
        original_filename=safe_name,
        csv_path=dest,
        source="upload",
    )
    return _dataset_out(ds)


@router.post("/datasets/load-sample", response_model=DatasetOut, status_code=201)
def load_sample(body: LoadSampleRequest, db: Session = Depends(get_db)):
    dataset_service.ensure_dirs()
    ds = dataset_service.load_sample(db, body.name.strip())
    return _dataset_out(ds)


@router.get("/datasets/samples", response_model=list[SampleInfo])
def list_samples():
    return dataset_service.list_samples()


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    rows = (
        db.query(Dataset).order_by(Dataset.created_at.desc()).limit(limit).all()
    )
    return [_dataset_out(r) for r in rows]


@router.get("/datasets/{dataset_id}", response_model=DatasetDetail)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = _get_ds(db, dataset_id)
    return {
        "dataset": _dataset_out(ds),
        "analysis": _analysis_out(ds.analysis) if ds.analysis else None,
        "models": _training_out(db, ds),
    }


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete the dataset, its uploaded file and its saved models."""
    return dataset_service.delete_dataset(db, dataset_id)


@router.post("/datasets/{dataset_id}/analyze", response_model=AnalysisOut)
def analyze(dataset_id: str, body: AnalyzeRequest | None = None, db: Session = Depends(get_db)):
    _get_ds(db, dataset_id)  # 404 check
    target = body.target if body else None
    a = analysis_service.run_analysis(db, dataset_id, target_override=target)
    return _analysis_out(a)


@router.post("/datasets/{dataset_id}/train", response_model=ModelRunOut)
def train(dataset_id: str, body: TrainRequest | None = None, db: Session = Depends(get_db)):
    ds = _get_ds(db, dataset_id)
    body = body or TrainRequest()
    k = (body.k, min(body.k + 6, 12)) if body.k else None
    result = ml_service.run_training(db, dataset_id, target=body.target,
                                     problem_type=body.problem_type, k_range=k,
                                     features=body.features,
                                     drop_outliers=body.drop_outliers,
                                     tune=body.tune,
                                     impute_numeric=body.impute_numeric,
                                     impute_categorical=body.impute_categorical,
                                     engineered=body.engineered,
                                     balance_classes=body.balance_classes)
    db.refresh(ds)
    return _training_out(db, ds)


@router.get("/datasets/{dataset_id}/models", response_model=ModelRunOut)
def get_models(dataset_id: str, db: Session = Depends(get_db)):
    ds = _get_ds(db, dataset_id)
    out = _training_out(db, ds)
    if out is None:
        raise ValidationError("No training run exists for this dataset yet.")
    return out


@router.get("/datasets/{dataset_id}/visualizations")
def get_visualizations(dataset_id: str, db: Session = Depends(get_db)):
    ds = _get_ds(db, dataset_id)
    if ds.analysis is None:
        raise ValidationError("Analyze the dataset first to generate visualizations.")
    return {"dataset_id": ds.id, "figures": ds.analysis.figures or []}


@router.post("/datasets/{dataset_id}/chat", response_model=ChatOut)
def chat(dataset_id: str, body: ChatRequest, db: Session = Depends(get_db)):
    ds = _get_ds(db, dataset_id)
    training = ml_service.training_from_db(db, ds)
    result = chat_service.answer_question(ds, ds.analysis, training, body.message.strip())
    return result


# ------------------------------------------------------------ analyst layer
# Ad-hoc analysis over the raw data, alongside the modelling pipeline: ask a
# question, run a structured query, follow a trend, compare segments, or let the
# agent surface what it finds on its own.
@router.get("/datasets/{dataset_id}/schema")
def get_schema(dataset_id: str, db: Session = Depends(get_db)):
    return to_jsonable(analyst_service.schema(db, dataset_id))


@router.get("/datasets/{dataset_id}/suggested-questions")
def get_suggested_questions(dataset_id: str, db: Session = Depends(get_db)):
    return {"questions": analyst_service.suggested_questions(db, dataset_id)}


@router.post("/datasets/{dataset_id}/ask", response_model=AskOut)
def ask(dataset_id: str, body: AskRequest, db: Session = Depends(get_db)):
    return to_jsonable(analyst_service.ask(db, dataset_id, body.question.strip()))


@router.post("/datasets/{dataset_id}/query")
def run_structured_query(dataset_id: str, body: QuerySpec, db: Session = Depends(get_db)):
    return to_jsonable(analyst_service.query(db, dataset_id, body))


@router.post("/datasets/{dataset_id}/pivot")
def run_pivot(dataset_id: str, body: PivotRequest, db: Session = Depends(get_db)):
    return to_jsonable(
        analyst_service.pivot(db, dataset_id, body.index, body.columns, body.values, body.aggfunc)
    )


@router.post("/datasets/{dataset_id}/trend")
def get_trend(dataset_id: str, body: TrendRequest | None = None, db: Session = Depends(get_db)):
    body = body or TrendRequest()
    return to_jsonable(
        analyst_service.trend(db, dataset_id, body.date_column, body.value_column,
                              agg=body.agg, freq=body.freq)
    )


@router.post("/datasets/{dataset_id}/segments")
def get_segments(dataset_id: str, body: SegmentRequest, db: Session = Depends(get_db)):
    return to_jsonable(analyst_service.segments(db, dataset_id, body.dimension, body.metric))


@router.get("/datasets/{dataset_id}/predict/schema")
def predict_schema(dataset_id: str, db: Session = Depends(get_db)):
    """Which columns a prediction needs, plus an example row."""
    return predict_service.required_inputs(db, dataset_id)


@router.post("/datasets/{dataset_id}/predict")
def predict(dataset_id: str, body: PredictRequest, db: Session = Depends(get_db)):
    return predict_service.predict(db, dataset_id, body.rows, body.model_name)


@router.post("/datasets/{dataset_id}/predict/csv")
async def predict_csv(dataset_id: str, file: UploadFile = File(...),
                      db: Session = Depends(get_db)):
    from ..core.security import read_validated_csv

    content, _name = await read_validated_csv(file)
    return predict_service.predict_csv(db, dataset_id, content)


@router.get("/datasets/{dataset_id}/progress")
def progress_endpoint(dataset_id: str, db: Session = Depends(get_db)):
    from ..agents import progress as progress_tracker

    _get_ds(db, dataset_id)
    return progress_tracker.snapshot(dataset_id)


@router.post("/datasets/{dataset_id}/improve")
def improve_endpoint(dataset_id: str, body: ImproveRequest | None = None,
                     db: Session = Depends(get_db)):
    body = body or ImproveRequest()
    return to_jsonable(
        analyst_service.improvements(db, dataset_id, body.target, body.problem_type)
    )


@router.post("/datasets/{dataset_id}/dashboard")
def dashboard_endpoint(dataset_id: str, body: DashboardRequest | None = None,
                       db: Session = Depends(get_db)):
    body = body or DashboardRequest()
    filters = [Filter.model_validate(f) for f in body.filters]
    return to_jsonable(
        analyst_service.dashboard(db, dataset_id, filters, body.measure,
                                  body.dimension, body.date_column)
    )


@router.post("/datasets/{dataset_id}/suggest-features")
def suggest_features_endpoint(dataset_id: str, body: SuggestFeaturesRequest | None = None,
                              db: Session = Depends(get_db)):
    body = body or SuggestFeaturesRequest()
    return to_jsonable(
        analyst_service.feature_suggestion(db, dataset_id, body.target, body.problem_type)
    )


@router.get("/datasets/{dataset_id}/insights", response_model=InsightsOut)
def get_insights(dataset_id: str, limit: int = Query(12, ge=1, le=40), db: Session = Depends(get_db)):
    return to_jsonable(analyst_service.insights(db, dataset_id, limit=limit))


@router.post("/datasets/{dataset_id}/report", response_model=ReportOut)
def create_report(dataset_id: str, db: Session = Depends(get_db)):
    _get_ds(db, dataset_id)
    report = report_service.run_report(db, dataset_id)
    return {
        "id": report.id,
        "dataset_id": report.dataset_id,
        "format": report.format,
        "created_at": report.created_at,
        "content": report_service.report_content(report),
        "download_url": f"/api/reports/{report.id}/download",
    }


@router.get("/datasets/{dataset_id}/models/{name}/download")
def download_model(dataset_id: str, name: str, db: Session = Depends(get_db)):
    """Download a trained model artifact (.pkl, joblib) by its model name slug."""
    ds = _get_ds(db, dataset_id)
    settings = get_settings()
    path = safe_path(Path(settings.model_dir), ds.id, f"{name}.pkl")
    if not path.exists():
        raise ValidationError(
            "Model file not found. Run training again to regenerate model artifacts."
        )
    return FileResponse(
        path, media_type="application/octet-stream", filename=f"{name}.pkl"
    )


@router.get("/datasets/{dataset_id}/models/{name}/metadata")
def download_model_metadata(dataset_id: str, name: str, db: Session = Depends(get_db)):
    """Metadata (features, class labels, preprocessing notes) for the run's models."""
    ds = _get_ds(db, dataset_id)
    settings = get_settings()
    path = safe_path(Path(settings.model_dir), ds.id, "metadata.json")
    if not path.exists():
        raise ValidationError("Model metadata not found. Run training again first.")
    return FileResponse(path, media_type="application/json", filename="metadata.json")


@router.get("/reports/{report_id}/download")
def download_report(report_id: str, db: Session = Depends(get_db)):
    report = db.get(Report, report_id)
    if report is None:
        raise ReportNotFoundError("Report not found.")
    path = Path(report.content_path)
    if not path.exists():
        raise ReportNotFoundError("The report file is missing on disk.")
    filename = path.name
    return FileResponse(path, media_type="text/markdown", filename=filename)
