"""Pydantic response/request schemas (API contract)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ProblemType = Literal["regression", "classification", "clustering"]


# ---------------------------------------------------------------- datasets
class DatasetOut(BaseModel):
    id: str
    name: str
    original_filename: str
    rows: int
    columns: int
    column_names: list[str]
    source: str
    status: str
    created_at: datetime
    has_analysis: bool = False
    has_training: bool = False
    best_model: str | None = None
    problem_type: ProblemType | None = None
    target: str | None = None


class DatasetDetail(BaseModel):
    dataset: DatasetOut
    analysis: "AnalysisOut | None" = None
    models: "ModelRunOut | None" = None


class SampleInfo(BaseModel):
    name: str
    label: str
    description: str
    rows: int
    columns: int
    task: str


class LoadSampleRequest(BaseModel):
    name: str = Field(..., description="Sample dataset name (see /api/datasets/samples)")


# ---------------------------------------------------------------- analysis
class FigureOut(BaseModel):
    id: str
    kind: str
    title: str
    data: dict[str, Any]  # Plotly figure (data + layout)


class TraceStep(BaseModel):
    tool: str
    args: dict[str, Any]
    status: str
    observation: str
    duration_s: float


class AnalyzeRequest(BaseModel):
    target: str | None = Field(None, description="Override the auto-detected target column")


class AnalysisOut(BaseModel):
    id: str
    dataset_id: str
    target_column: str | None
    problem_type: ProblemType
    target_candidates: list[dict[str, Any]] = []
    target_detection: dict[str, Any] = {}
    profile: dict[str, Any]
    missing: dict[str, Any]
    statistics: dict[str, Any]
    correlation: dict[str, Any] | None
    outliers: dict[str, Any]
    preview: dict[str, Any]
    figures: list[FigureOut]
    plan: list[str]
    trace: list[TraceStep]
    explanation: str
    created_at: datetime


# ---------------------------------------------------------------- training
class TrainRequest(BaseModel):
    target: str | None = Field(None, description="Target column (auto-detected if omitted)")
    problem_type: ProblemType | None = Field(None, description="Override task type")
    k: int | None = Field(None, ge=2, le=12, description="Clusters for K-Means (auto-search if omitted)")
    features: list[str] | None = Field(
        None,
        max_length=500,
        description="Input columns to train on. Omit to use every column except the target.",
    )
    drop_outliers: bool = Field(
        False, description="Remove rows outside 1.5xIQR on a numeric column before training"
    )


class ModelResultOut(BaseModel):
    name: str
    model_type: str
    status: str
    status_message: str = ""
    primary_metric: float | None
    metrics: dict[str, Any]
    feature_importance: list[dict[str, Any]] | None
    training_seconds: float
    is_best: bool
    # 1 = best by the task's primary metric. Without this field the response
    # model silently drops it and the table renders in registry order.
    rank: int | None = None
    download_url: str | None = None


class ModelRunOut(BaseModel):
    dataset_id: str
    run_id: str
    problem_type: ProblemType
    target: str | None
    run_info: dict[str, Any]
    features_used: list[str] | None = None
    source_columns: list[str] = []
    drop_outliers: bool = False
    models: list[ModelResultOut]
    best_model: str | None
    explanation: str
    trace: list[TraceStep]
    created_at: datetime


# ---------------------------------------------------------------- report & chat
class ReportOut(BaseModel):
    id: str
    dataset_id: str
    format: str
    created_at: datetime
    content: str
    download_url: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatOut(BaseModel):
    reply: str
    tool_calls: list[str]
    source: str  # "llm" | "local-engine"


# ---------------------------------------------------------------- analyst
AggFnName = Literal["count", "sum", "mean", "median", "min", "max", "std", "nunique"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class TrendRequest(BaseModel):
    date_column: str | None = None
    value_column: str | None = None
    agg: AggFnName = "sum"
    freq: str | None = Field(None, pattern=r"^(D|W|ME|QE|YE)$")


class SegmentRequest(BaseModel):
    dimension: str = Field(..., min_length=1, max_length=200)
    metric: str | None = Field(None, max_length=200)


class PivotRequest(BaseModel):
    index: str = Field(..., min_length=1, max_length=200)
    columns: str = Field(..., min_length=1, max_length=200)
    values: str = Field(..., min_length=1, max_length=200)
    aggfunc: AggFnName = "sum"


class InsightOut(BaseModel):
    kind: str
    title: str
    detail: str
    importance: float
    columns: list[str]
    evidence: dict[str, Any]


class InsightsOut(BaseModel):
    dataset_id: str
    count: int
    returned: int
    headline: str
    by_kind: dict[str, int]
    insights: list[InsightOut]
    tools_used: list[str]


class AskOut(BaseModel):
    question: str
    route: str  # "llm" | "heuristic"
    answer: str
    table: dict[str, Any]
    chart: dict[str, Any] | None = None
    tools_used: list[str]


class SuggestFeaturesRequest(BaseModel):
    target: str | None = None
    problem_type: ProblemType | None = None


class DashboardRequest(BaseModel):
    filters: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    measure: str | None = None
    dimension: str | None = None
    date_column: str | None = None


class ImproveRequest(BaseModel):
    target: str | None = None
    problem_type: ProblemType | None = None
