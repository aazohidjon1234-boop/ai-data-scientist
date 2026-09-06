"""Analyst orchestration: ask, query, time series, segments, insights.

Thin layer between the API and the tools — loads the dataset's DataFrame and
delegates. No maths lives here.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from ..agents.analyst import answer_question
from ..agents.dashboard import build_dashboard
from ..agents.feature_advisor import suggest_features
from ..agents.insights import generate_insights
from ..agents.llm_client import LLMClient
from ..exceptions import ValidationError
from ..models.db import Dataset
from ..tools.analytics_tools import compare_segments, detect_datetime_columns, time_series
from ..tools.query_tools import (
    MAX_DIMENSION_SUGGESTIONS,
    Filter,
    QuerySpec,
    describe_schema,
    pivot_table,
    run_query,
)
from ..utils.dataframe_store import load_csv_cached
from . import dataset_service


def _frame(db: Session, dataset_id: str) -> tuple[Dataset, pd.DataFrame]:
    ds = dataset_service.get_dataset_or_404(db, dataset_id)
    return ds, load_csv_cached(ds.file_path)


def schema(db: Session, dataset_id: str) -> dict[str, Any]:
    """Columns, roles and the dimensions/measures worth offering in the UI."""
    ds, df = _frame(db, dataset_id)
    described = describe_schema(df)
    dimensions = [
        c["name"] for c in described["columns"]
        if c["kind"] in {"categorical", "boolean"} or c["unique"] <= MAX_DIMENSION_SUGGESTIONS
    ]
    measures = [c["name"] for c in described["columns"] if c["kind"] == "numeric"]
    return {
        "dataset_id": ds.id,
        **described,
        "dimensions": dimensions,
        "measures": measures,
        "date_columns": detect_datetime_columns(df),
    }


def ask(db: Session, dataset_id: str, question: str) -> dict[str, Any]:
    _, df = _frame(db, dataset_id)
    return answer_question(df, question, LLMClient())


def query(db: Session, dataset_id: str, spec: QuerySpec) -> dict[str, Any]:
    _, df = _frame(db, dataset_id)
    return run_query(df, spec)


def pivot(db: Session, dataset_id: str, index: str, columns: str,
          values: str, aggfunc: str = "sum") -> dict[str, Any]:
    _, df = _frame(db, dataset_id)
    return pivot_table(df, index, columns, values, aggfunc)  # type: ignore[arg-type]


def trend(db: Session, dataset_id: str, date_column: str | None,
          value_column: str | None, agg: str = "sum",
          freq: str | None = None) -> dict[str, Any]:
    _, df = _frame(db, dataset_id)
    if not date_column:
        found = detect_datetime_columns(df)
        if not found:
            raise ValidationError(
                "No date column was found in this dataset, so there is no timeline to analyse."
            )
        date_column = found[0]
    return time_series(df, date_column, value_column, agg=agg, freq=freq)


def segments(db: Session, dataset_id: str, dimension: str,
             metric: str | None = None) -> dict[str, Any]:
    _, df = _frame(db, dataset_id)
    return compare_segments(df, dimension, metric)


def insights(db: Session, dataset_id: str, limit: int = 12) -> dict[str, Any]:
    ds, df = _frame(db, dataset_id)
    result = generate_insights(df, limit=limit)
    result["dataset_id"] = ds.id
    return result


def suggested_questions(db: Session, dataset_id: str, limit: int = 6) -> list[str]:
    """Starter questions built from this dataset's real columns."""
    _, df = _frame(db, dataset_id)
    described = describe_schema(df)
    dimensions = [c["name"] for c in described["columns"]
                  if c["kind"] in {"categorical", "boolean"} and 2 <= c["unique"] <= 15]
    measures = [c["name"] for c in described["columns"]
                if c["kind"] == "numeric" and c["unique"] > 15]
    dates = detect_datetime_columns(df)

    questions: list[str] = []
    if dimensions and measures:
        questions.append(f"Which {dimensions[0]} has the highest average {measures[0]}?")
        if len(measures) > 1:
            questions.append(f"What is the total {measures[1]} by {dimensions[0]}?")
        if len(dimensions) > 1:
            questions.append(f"Compare {measures[0]} across {dimensions[1]}")
    if dimensions:
        questions.append(f"How many rows are in each {dimensions[0]}?")
    if dates and measures:
        questions.append(f"How has {measures[0]} changed over time?")
    if measures:
        questions.append(f"What is the average {measures[0]}?")
    return questions[:limit]


def feature_suggestion(db: Session, dataset_id: str, target: str | None,
                       problem_type: str | None) -> dict[str, Any]:
    ds, df = _frame(db, dataset_id)
    chosen = target or (ds.analysis.target_column if ds.analysis else None)
    kind = problem_type or (ds.analysis.problem_type if ds.analysis else None) or "classification"
    if not chosen:
        raise ValidationError("Pick a target column first — suggestions are relative to it.")
    return suggest_features(df, chosen, kind)


def dashboard(db: Session, dataset_id: str, filters: list[Filter] | None,
              measure: str | None, dimension: str | None,
              date_column: str | None) -> dict[str, Any]:
    ds, df = _frame(db, dataset_id)
    result = build_dashboard(df, filters, measure, dimension, date_column)
    result["dataset_id"] = ds.id
    result["dataset_name"] = ds.name
    return result
