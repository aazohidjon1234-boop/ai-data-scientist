"""Auto-composed dashboard: KPI tiles, charts and filters for any CSV.

The datasets here are arbitrary, so the layout cannot be hand-drawn. It is
derived from the schema instead: numeric columns that vary become measures,
low-cardinality columns become dimensions and filters, and a date column (when
one exists) becomes the timeline.

Filters run through the same validated query path as everything else, so a
dashboard cannot ask for a column that is not there, and every number on it was
computed by pandas.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..exceptions import ValidationError
from ..tools.analytics_tools import detect_datetime_columns, as_datetime, _auto_freq
from ..tools.query_tools import Filter, apply_filters, describe_schema
from ..tools.viz_tools import ACCENT, PALETTE, _wrap
from ..utils.jsonutils import to_jsonable

MAX_FILTER_VALUES = 25
MAX_DIMENSION_CARDINALITY = 20
MAX_BARS = 12


# ---------------------------------------------------------------- layout
def _is_identifier(series: pd.Series, rows: int) -> bool:
    unique = series.nunique(dropna=True)
    if unique / max(rows, 1) < 0.95 or unique < 20:
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return True
    values = series.dropna()
    if values.empty or not np.all(np.equal(np.mod(values.to_numpy(dtype=float), 1), 0)):
        return False
    return bool(float(values.max() - values.min()) + 1 <= unique * 1.5)


def dashboard_layout(df: pd.DataFrame) -> dict[str, Any]:
    """Which columns are worth putting on a dashboard, and as what."""
    rows = len(df)
    measures, dimensions = [], []
    for name in df.columns:
        series = df[name]
        if _is_identifier(series, rows):
            continue
        unique = series.nunique(dropna=True)
        if unique <= 1:
            continue
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if unique > MAX_DIMENSION_CARDINALITY:
                measures.append(str(name))
            else:
                dimensions.append(str(name))  # ratings, counts: better as groups
        elif unique <= MAX_DIMENSION_CARDINALITY:
            dimensions.append(str(name))

    dates = detect_datetime_columns(df)
    return {
        "measures": measures,
        "dimensions": dimensions,
        "date_columns": dates,
        "primary_measure": measures[0] if measures else None,
        "primary_dimension": dimensions[0] if dimensions else None,
        "primary_date": dates[0] if dates else None,
    }


def filter_options(df: pd.DataFrame, dimensions: list[str]) -> list[dict[str, Any]]:
    options = []
    for name in dimensions:
        counts = df[name].value_counts(dropna=True).head(MAX_FILTER_VALUES)
        options.append({
            "column": name,
            "values": [str(v) for v in counts.index],
            "truncated": bool(df[name].nunique(dropna=True) > MAX_FILTER_VALUES),
        })
    return options


# ------------------------------------------------------------------ tiles
def _fmt_hint(series: pd.Series) -> str:
    return f"{series.notna().sum():,} values"


def _kpis(df: pd.DataFrame, full: pd.DataFrame, measure: str | None,
          dimension: str | None, date_column: str | None) -> list[dict[str, Any]]:
    tiles: list[dict[str, Any]] = [{
        "label": "Rows",
        "value": len(df),
        "format": "int",
        "hint": f"of {len(full):,} total" if len(df) != len(full) else "no filters applied",
    }]
    if measure and measure in df.columns and df[measure].notna().any():
        column = df[measure]
        tiles.append({"label": f"Total {measure}", "value": float(column.sum()),
                      "format": "num", "hint": _fmt_hint(column)})
        tiles.append({"label": f"Average {measure}", "value": float(column.mean()),
                      "format": "num", "hint": f"median {column.median():,.2f}"})
    if dimension and dimension in df.columns and not df.empty:
        counts = df[dimension].value_counts(dropna=True)
        if not counts.empty:
            share = 100.0 * counts.iloc[0] / counts.sum()
            tiles.append({"label": f"Top {dimension}", "value": str(counts.index[0]),
                          "format": "text", "hint": f"{share:.0f}% of rows"})
    if date_column and date_column in df.columns and not df.empty:
        try:
            when = as_datetime(df, date_column).dropna()
            if not when.empty:
                tiles.append({
                    "label": "Date range", "format": "text",
                    "value": f"{when.min().date()} → {when.max().date()}",
                    "hint": f"{(when.max() - when.min()).days:,} days",
                })
        except ValidationError:
            pass
    return tiles


# ----------------------------------------------------------------- charts
def _bar_by_dimension(df: pd.DataFrame, dimension: str, measure: str | None) -> dict[str, Any] | None:
    if dimension not in df.columns or df.empty:
        return None
    if measure and measure in df.columns:
        grouped = df.groupby(dimension, dropna=False, observed=True)[measure].mean()
        title, axis = f"Average {measure} by {dimension}", f"avg {measure}"
    else:
        grouped = df[dimension].value_counts(dropna=False)
        title, axis = f"Rows by {dimension}", "rows"
    grouped = grouped.sort_values(ascending=False).head(MAX_BARS)
    if grouped.empty:
        return None
    fig = go.Figure(go.Bar(x=[str(i) for i in grouped.index],
                           y=[float(v) for v in grouped.to_numpy()], marker_color=ACCENT))
    fig.update_layout(xaxis_title=dimension, yaxis_title=axis)
    return _wrap(fig, "bar", title)


def _share_donut(df: pd.DataFrame, dimension: str) -> dict[str, Any] | None:
    if dimension not in df.columns or df.empty:
        return None
    counts = df[dimension].value_counts(dropna=False).head(MAX_BARS)
    if counts.empty:
        return None
    fig = go.Figure(go.Pie(labels=[str(i) for i in counts.index],
                           values=[int(v) for v in counts.to_numpy()],
                           hole=0.55, marker=dict(colors=PALETTE)))
    return _wrap(fig, "pie", f"Share of {dimension}")


def _distribution(df: pd.DataFrame, measure: str) -> dict[str, Any] | None:
    if measure not in df.columns:
        return None
    values = df[measure].dropna()
    if values.empty:
        return None
    fig = go.Figure(go.Histogram(x=values.to_numpy(dtype=float), nbinsx=30, marker_color=ACCENT))
    fig.update_layout(xaxis_title=measure, yaxis_title="rows")
    return _wrap(fig, "histogram", f"Distribution of {measure}")


def _timeline(df: pd.DataFrame, date_column: str, measure: str | None) -> dict[str, Any] | None:
    if df.empty:
        return None
    try:
        when = as_datetime(df, date_column)
    except ValidationError:
        return None
    frame = pd.DataFrame({"_when": when})
    frame["_value"] = df[measure].to_numpy() if measure and measure in df.columns else 1.0
    how = "mean" if measure else "sum"
    frame = frame.dropna(subset=["_when"]).set_index("_when").sort_index()
    if len(frame) < 2:
        return None
    series = frame["_value"].resample(_auto_freq(frame.index)).agg(how).dropna()
    if len(series) < 2:
        return None
    fig = go.Figure(go.Scatter(x=[str(i.date()) for i in series.index],
                               y=[float(v) for v in series.to_numpy()],
                               mode="lines", line=dict(color=ACCENT, width=2)))
    fig.update_layout(xaxis_title=date_column,
                      yaxis_title=f"{how} {measure}" if measure else "rows")
    return _wrap(fig, "line", f"{measure or 'Rows'} over time")


def _cross_heatmap(df: pd.DataFrame, a: str, b: str, measure: str | None) -> dict[str, Any] | None:
    if a == b or a not in df.columns or b not in df.columns or df.empty:
        return None
    if measure and measure in df.columns:
        table = pd.pivot_table(df, index=a, columns=b, values=measure, aggfunc="mean")
        title = f"Average {measure}: {a} × {b}"
    else:
        table = pd.crosstab(df[a], df[b])
        title = f"Row counts: {a} × {b}"
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return None
    table = table.iloc[:MAX_BARS, :MAX_BARS]
    fig = go.Figure(go.Heatmap(
        z=[[None if pd.isna(v) else float(v) for v in row] for row in table.to_numpy()],
        x=[str(c) for c in table.columns], y=[str(i) for i in table.index],
        # Custom ramp to the app's accent; Plotly has no "Indigo" scale.
        colorscale=[[0.0, "#eef2ff"], [0.5, "#a5b4fc"], [1.0, ACCENT]],
        hoverongaps=False))
    fig.update_layout(xaxis_title=b, yaxis_title=a)
    return _wrap(fig, "heatmap", title)


# ------------------------------------------------------------------ entry
def build_dashboard(
    df: pd.DataFrame,
    filters: list[Filter] | None = None,
    measure: str | None = None,
    dimension: str | None = None,
    date_column: str | None = None,
) -> dict[str, Any]:
    layout = dashboard_layout(df)
    measure = measure or layout["primary_measure"]
    dimension = dimension or layout["primary_dimension"]
    date_column = date_column or layout["primary_date"]

    for name, role in ((measure, "measure"), (dimension, "dimension"), (date_column, "date column")):
        if name is not None and name not in df.columns:
            raise ValidationError(f"Unknown {role} '{name}'.")

    filtered = apply_filters(df, filters or [])

    charts: list[dict[str, Any]] = []
    if date_column:
        charts.append(_timeline(filtered, date_column, measure))
    if dimension:
        charts.append(_bar_by_dimension(filtered, dimension, measure))
    if measure:
        charts.append(_distribution(filtered, measure))
    if dimension:
        charts.append(_share_donut(filtered, dimension))
    others = [d for d in layout["dimensions"] if d != dimension]
    if dimension and others:
        charts.append(_cross_heatmap(filtered, dimension, others[0], measure))

    return to_jsonable({
        "layout": layout,
        "measure": measure,
        "dimension": dimension,
        "date_column": date_column,
        "filters_applied": [f.model_dump() for f in (filters or [])],
        "rows_total": len(df),
        "rows_shown": len(filtered),
        "kpis": _kpis(filtered, df, measure, dimension, date_column),
        "charts": [c for c in charts if c],
        "filter_options": filter_options(df, layout["dimensions"]),
        "empty": filtered.empty,
        "schema": describe_schema(df),
        "tools_used": ["dashboard_layout", "run_query", "create_visualization"],
    })
