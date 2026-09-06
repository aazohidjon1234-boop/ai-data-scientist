"""tool: create_visualization — Plotly figure builders.

All figures are returned as plain JSON dicts (plotly `data` + `layout`)
and rendered in the frontend with react-plotly.js.
"""
from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core.config import get_settings
from ..utils.jsonutils import fig_to_jsonable

ACCENT = "#6366f1"
PALETTE = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4", "#a855f7", "#ec4899"]


def _wrap(fig: go.Figure, kind: str, title: str) -> dict[str, Any]:
    fig.update_layout(
        title=None,
        margin=dict(l=46, r=18, t=14, b=42),
        height=340,
        hovermode="x unified",
    )
    return {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "title": title,
        "data": fig_to_jsonable(fig),
    }


def _numeric(df: pd.DataFrame, limit: int = 6) -> list[str]:
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 2
    ][:limit]


def _categorical(df: pd.DataFrame, limit: int = 6) -> list[str]:
    return [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and 2 <= df[c].nunique(dropna=True) <= 15
    ][:limit]


def create_histograms(df: pd.DataFrame) -> list[dict[str, Any]]:
    figs = []
    for c in _numeric(df, 6):
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=df[c].dropna(), nbinsx=30, marker_color=ACCENT, opacity=0.85)
        )
        fig.update_layout(xaxis_title=str(c), yaxis_title="count")
        figs.append(_wrap(fig, "histogram", f"Distribution of {c}"))
    return figs


def create_box_plot(df: pd.DataFrame) -> dict[str, Any] | None:
    cols = _numeric(df, 6)
    if len(cols) < 2:
        return None
    # standardize so boxes are comparable across scales
    data = df[cols].copy()
    data = (data - data.mean()) / data.std(ddof=0).replace(0, 1)
    fig = go.Figure()
    for i, c in enumerate(cols):
        fig.add_trace(go.Box(y=data[c].dropna(), name=str(c)))
    fig.update_layout(yaxis_title="standardized value (z-score)", showlegend=True)
    return _wrap(fig, "boxplot", "Outlier check — numeric features (z-scores)")


def create_correlation_heatmap(corr: dict[str, Any]) -> dict[str, Any] | None:
    if corr is None:
        return None
    fig = go.Figure(
        go.Heatmap(
            z=corr["matrix"],
            x=corr["columns"],
            y=corr["columns"],
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            colorbar=dict(thickness=12),
            hovertemplate="%{x} × %{y}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _wrap(fig, "heatmap", "Correlation matrix (numeric columns)")


def create_scatter(df: pd.DataFrame, corr: dict[str, Any] | None, target: str | None) -> list[dict[str, Any]]:
    figs = []
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique(dropna=True) > 5:
        nums = [c for c in _numeric(df, 8) if c != target]
        pairs = []
        for c in nums:
            m = df[[c, target]].dropna()
            if len(m) > 2 and m[c].std() > 0 and m[target].std() > 0:
                v = float(np.corrcoef(m[c], m[target])[0, 1])
                pairs.append((abs(v), v, c))
        pairs.sort(reverse=True)
        for _, v, c in pairs[:2]:
            m = df[[c, target]].dropna()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=m[c], y=m[target], mode="markers",
                    marker=dict(size=6, color=ACCENT, opacity=0.6),
                    name=str(c),
                )
            )
            fig.update_layout(xaxis_title=str(c), yaxis_title=str(target))
            figs.append(_wrap(fig, "scatter", f"{c} vs {target} (r={v:.2f})"))
    elif corr:
        best = corr["top_pairs"][0]
        m = df[[best["a"], best["b"]]].dropna()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=m[best["a"]], y=m[best["b"]], mode="markers",
                marker=dict(size=6, color=ACCENT, opacity=0.6),
            )
        )
        fig.update_layout(xaxis_title=best["a"], yaxis_title=best["b"])
        figs.append(
            _wrap(fig, "scatter", f"{best['a']} vs {best['b']} (r={best['corr']:.2f})")
        )
    return figs


def create_bar_charts(df: pd.DataFrame, limit: int = 2) -> list[dict[str, Any]]:
    figs = []
    for c in _categorical(df, limit):
        vc = df[c].value_counts().head(12)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=vc.index.astype(str),
                y=vc.values,
                marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(vc))],
            )
        )
        fig.update_layout(xaxis_title=str(c), yaxis_title="count")
        figs.append(_wrap(fig, "bar", f"Values of {c}"))
    return figs


def create_target_distribution(df: pd.DataFrame, target: str | None) -> dict[str, Any] | None:
    if not target or target not in df.columns:
        return None
    s = df[target].dropna()
    fig = go.Figure()
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 8:
        fig.add_trace(go.Histogram(x=s, nbinsx=30, marker_color="#22c55e", opacity=0.85))
        fig.update_layout(xaxis_title=str(target), yaxis_title="count")
    else:
        vc = s.value_counts().head(12)
        fig.add_trace(
            go.Bar(x=vc.index.astype(str), y=vc.values, marker_color="#22c55e")
        )
        fig.update_layout(xaxis_title=str(target), yaxis_title="count")
    return _wrap(fig, "target-distribution", f"Target distribution — {target}")


def create_missing_bar(df: pd.DataFrame) -> dict[str, Any] | None:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return None
    rows = max(1, df.shape[0])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=missing.index.astype(str),
            y=(missing / rows * 100).round(2).values,
            marker_color="#f59e0b",
        )
    )
    fig.update_layout(xaxis_title="column", yaxis_title="% missing")
    return _wrap(fig, "missing", "Missing values by column (%)")


def create_clustering_preview(df: pd.DataFrame, labels: np.ndarray, feature_names: list[str]) -> dict[str, Any] | None:
    import plotly.express as px

    fig = px.scatter(
        df,
        x=feature_names[0],
        y=feature_names[1],
        color=pd.Series(labels, name="cluster"),
        color_discrete_sequence=PALETTE,
        title=None,
    )
    fig = fig.update_layout(legend_title_text="cluster")
    return _wrap(fig, "clusters", "Cluster preview (first two numeric features)")
