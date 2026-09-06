"""Ad-hoc querying — the data-analyst half of the agent.

A question like "which region had the highest average sales?" becomes a
**validated QuerySpec**, never generated Python. The spec is a small, closed
grammar (filter / group_by / aggregate / sort / limit); pandas executes it.

Why not let the model write pandas or SQL and run it? Because `eval`/`exec` on
model output is arbitrary code execution against user-uploaded data, and it
would break this project's rule that uploads never execute anything. A closed
grammar keeps the same guarantee as the rest of the tools: the model chooses
*what* to compute, Python decides *whether that is even expressible*, and every
number in the answer came out of pandas.

Unknown columns are rejected with the list of real ones, so a hallucinated
column name fails loudly instead of silently answering about nothing.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from ..exceptions import ValidationError
from ..utils.jsonutils import to_jsonable

MAX_LIMIT = 1000
MAX_GROUPS = 500
SAMPLE_VALUES = 5
# A numeric column with at most this many distinct values still reads as a
# dimension to group by (ratings, bedroom counts) rather than a measure.
MAX_DIMENSION_SUGGESTIONS = 15

FilterOp = Literal[
    "=", "!=", ">", ">=", "<", "<=",
    "in", "not_in", "contains", "between", "is_null", "not_null",
]
AggFn = Literal["count", "sum", "mean", "median", "min", "max", "std", "nunique"]

# Aggregations that only make sense on numbers.
_NUMERIC_ONLY = {"sum", "mean", "median", "std"}


class Filter(BaseModel):
    column: str
    op: FilterOp = "="
    value: Any = None


class Aggregation(BaseModel):
    # `column` may be omitted only for count, which counts rows.
    column: str | None = None
    fn: AggFn = "count"
    alias: str | None = None

    def label(self) -> str:
        if self.alias:
            return self.alias
        return f"{self.fn}_{self.column}" if self.column else "row_count"


class QuerySpec(BaseModel):
    """The complete action space for an ad-hoc question."""

    filters: list[Filter] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    aggregations: list[Aggregation] = Field(default_factory=list)
    select: list[str] = Field(default_factory=list)
    sort_by: str | None = None
    ascending: bool = False
    limit: int = 25

    @field_validator("limit")
    @classmethod
    def _cap_limit(cls, v: int) -> int:
        return max(1, min(int(v), MAX_LIMIT))


# ---------------------------------------------------------------------------
# schema description (what the model is allowed to reference)
# ---------------------------------------------------------------------------

def describe_schema(df: pd.DataFrame, sample_values: int = SAMPLE_VALUES) -> dict[str, Any]:
    """Compact, safe column catalogue for prompting and for the UI."""
    columns = []
    for name in df.columns:
        series = df[name]
        kind = (
            "numeric" if pd.api.types.is_numeric_dtype(series)
            else "datetime" if pd.api.types.is_datetime64_any_dtype(series)
            else "boolean" if pd.api.types.is_bool_dtype(series)
            else "categorical"
        )
        entry: dict[str, Any] = {
            "name": str(name),
            "kind": kind,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if kind == "numeric" and series.notna().any():
            entry["min"] = to_jsonable(series.min())
            entry["max"] = to_jsonable(series.max())
        elif kind in {"categorical", "boolean"}:
            top = series.dropna().astype(str).value_counts().head(sample_values)
            entry["examples"] = [str(v) for v in top.index.tolist()]
        columns.append(entry)
    return {"rows": int(len(df)), "columns": columns}


def _require_column(df: pd.DataFrame, name: str, role: str) -> str:
    if name in df.columns:
        return name
    # Case-insensitive rescue: models frequently change capitalisation.
    lowered = {str(c).lower(): c for c in df.columns}
    if str(name).lower() in lowered:
        return lowered[str(name).lower()]
    raise ValidationError(
        f"Unknown column '{name}' in {role}. Available columns: "
        + ", ".join(str(c) for c in df.columns[:40])
    )


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------

def _coerce(series: pd.Series, value: Any) -> Any:
    """Make a JSON value comparable with the column it is filtered against."""
    if value is None:
        return None
    if pd.api.types.is_numeric_dtype(series) and not isinstance(value, (list, tuple)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if pd.api.types.is_datetime64_any_dtype(series) and not isinstance(value, (list, tuple)):
        parsed = pd.to_datetime(value, errors="coerce")
        return value if pd.isna(parsed) else parsed
    return value


def _mask(df: pd.DataFrame, f: Filter) -> pd.Series:
    column = _require_column(df, f.column, "filters")
    series = df[column]
    op, value = f.op, f.value

    if op == "is_null":
        return series.isna()
    if op == "not_null":
        return series.notna()
    if op in {"in", "not_in"}:
        values = value if isinstance(value, (list, tuple, set)) else [value]
        coerced = [_coerce(series, v) for v in values]
        if not pd.api.types.is_numeric_dtype(series):
            hit = series.astype(str).isin([str(v) for v in coerced])
        else:
            hit = series.isin(coerced)
        return ~hit if op == "not_in" else hit
    if op == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValidationError("'between' needs exactly two values: [low, high]")
        low, high = (_coerce(series, v) for v in value)
        return series.between(low, high)
    if op == "contains":
        return series.astype(str).str.contains(str(value), case=False, na=False)

    target = _coerce(series, value)
    if op in {">", ">=", "<", "<="} and not (
        pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)
    ):
        raise ValidationError(f"Cannot use '{op}' on non-numeric column '{column}'")
    if op == "=":
        return series.astype(str) == str(target) if not pd.api.types.is_numeric_dtype(series) else series == target
    if op == "!=":
        return series.astype(str) != str(target) if not pd.api.types.is_numeric_dtype(series) else series != target
    return {">": series.gt, ">=": series.ge, "<": series.lt, "<=": series.le}[op](target)


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for f in filters:
        mask &= _mask(df, f)
    return df[mask]


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def _aggregate(frame: pd.DataFrame, aggs: list[Aggregation], group_by: list[str]) -> pd.DataFrame:
    series_list: list[pd.Series] = []

    if group_by:
        grouped = frame.groupby(group_by, dropna=False, observed=True)
        for agg in aggs:
            label = agg.label()
            if agg.fn == "count" and not agg.column:
                series_list.append(grouped.size().rename(label))
            else:
                column = _require_column(frame, agg.column, "aggregations")
                series_list.append(grouped[column].agg(agg.fn).rename(label))
        return pd.concat(series_list, axis=1).reset_index()

    row: dict[str, Any] = {}
    for agg in aggs:
        if agg.fn == "count" and not agg.column:
            row[agg.label()] = len(frame)
        else:
            column = _require_column(frame, agg.column, "aggregations")
            row[agg.label()] = frame[column].agg(agg.fn)
    return pd.DataFrame([row])


def _validate_aggregations(df: pd.DataFrame, aggs: list[Aggregation]) -> None:
    for agg in aggs:
        if agg.fn == "count" and not agg.column:
            continue
        if not agg.column:
            raise ValidationError(f"Aggregation '{agg.fn}' needs a column")
        column = _require_column(df, agg.column, "aggregations")
        if agg.fn in _NUMERIC_ONLY and not pd.api.types.is_numeric_dtype(df[column]):
            raise ValidationError(
                f"Cannot compute {agg.fn} of '{column}' — it is not numeric. "
                "Use count or nunique for text columns."
            )


def run_query(df: pd.DataFrame, spec: QuerySpec) -> dict[str, Any]:
    """Execute a validated spec. Returns a JSON-safe table plus what it did."""
    group_by = [_require_column(df, c, "group_by") for c in spec.group_by]
    _validate_aggregations(df, spec.aggregations)

    filtered = apply_filters(df, spec.filters)
    rows_after_filter = int(len(filtered))
    if rows_after_filter == 0:
        return {
            "columns": [], "rows": [], "row_count": 0, "rows_after_filter": 0,
            "truncated": False, "spec": spec.model_dump(),
            "note": "No rows match these filters.",
        }

    if group_by and len(filtered.groupby(group_by, dropna=False, observed=True)) > MAX_GROUPS:
        raise ValidationError(
            f"Grouping by {', '.join(group_by)} produces more than {MAX_GROUPS} groups. "
            "Group by a column with fewer distinct values, or add a filter."
        )

    if spec.aggregations:
        result = _aggregate(filtered, spec.aggregations, group_by)
    elif group_by:
        result = _aggregate(filtered, [Aggregation(fn="count")], group_by)
    else:
        columns = [_require_column(df, c, "select") for c in spec.select] or list(filtered.columns)
        result = filtered[columns]

    if spec.sort_by:
        sort_column = _require_column(result, spec.sort_by, "sort_by")
        result = result.sort_values(sort_column, ascending=spec.ascending, kind="mergesort")
    elif spec.aggregations and group_by:
        # Most questions are "which group is biggest" — order by the first metric.
        result = result.sort_values(spec.aggregations[0].label(), ascending=False, kind="mergesort")

    total = int(len(result))
    result = result.head(spec.limit)

    return {
        "columns": [str(c) for c in result.columns],
        "rows": [
            {str(k): to_jsonable(v) for k, v in record.items()}
            for record in result.to_dict(orient="records")
        ],
        "row_count": total,
        "rows_after_filter": rows_after_filter,
        "truncated": total > spec.limit,
        "spec": spec.model_dump(),
    }


def pivot_table(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: AggFn = "sum",
) -> dict[str, Any]:
    """Cross-tab of two dimensions — the analyst's pivot table."""
    index = _require_column(df, index, "index")
    columns = _require_column(df, columns, "columns")
    values = _require_column(df, values, "values")
    if aggfunc in _NUMERIC_ONLY and not pd.api.types.is_numeric_dtype(df[values]):
        raise ValidationError(f"Cannot compute {aggfunc} of non-numeric column '{values}'")
    if df[index].nunique() * df[columns].nunique() > MAX_GROUPS * 4:
        raise ValidationError(
            f"'{index}' x '{columns}' is too large to pivot. Pick lower-cardinality columns."
        )

    table = pd.pivot_table(
        df, index=index, columns=columns, values=values, aggfunc=aggfunc, dropna=False
    )
    table = table.replace({np.nan: None})
    return {
        "index": str(index),
        "columns_field": str(columns),
        "values_field": str(values),
        "aggfunc": aggfunc,
        "column_labels": [str(c) for c in table.columns],
        "rows": [
            {"label": str(label), "values": [to_jsonable(v) for v in row]}
            for label, row in zip(table.index, table.to_numpy())
        ],
    }
