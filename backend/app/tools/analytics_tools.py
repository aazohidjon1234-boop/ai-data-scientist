"""Analyst tools: time series and segment comparison.

The modelling half of this project answers "what predicts the target?". These
answer the two questions an analyst is actually asked first:

* **Is it moving?** — trend, period-over-period growth, seasonality, anomalies.
* **Is the difference real?** — group comparison with a significance test, so
  "region A beats region B" is reported with a p-value instead of eyeballed off
  a bar chart.

Everything is computed with pandas/scipy and returned as plain JSON.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import ValidationError
from ..utils.jsonutils import to_jsonable

ALPHA = 0.05
MIN_SERIES_POINTS = 4
MAX_SEGMENTS = 30
_DATE_NAME_HINTS = ("date", "time", "day", "month", "year", "created", "updated",
                    "timestamp", "period", "when", "at")


# ---------------------------------------------------------------------------
# date detection
# ---------------------------------------------------------------------------

def detect_datetime_columns(df: pd.DataFrame, min_success: float = 0.8) -> list[str]:
    """Columns that are dates, or text that parses as dates almost always.

    Requires a high parse rate so a column of free text with one date-looking
    value is not mistaken for a timeline.
    """
    found: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            found.append(str(column))
            continue
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        sample = series.dropna().astype(str).head(200)
        if len(sample) < MIN_SERIES_POINTS:
            continue
        looks_dateish = any(h in str(column).lower() for h in _DATE_NAME_HINTS)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        rate = float(parsed.notna().mean())
        # Without a date-like name, demand near-perfect parsing.
        if rate >= (min_success if looks_dateish else 0.95):
            found.append(str(column))
    return found


def as_datetime(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValidationError(f"Unknown column '{column}'")
    series = df[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.notna().sum() < MIN_SERIES_POINTS:
        raise ValidationError(f"Column '{column}' does not contain usable dates")
    return parsed


MAX_SERIES_POINTS = 400


def _auto_freq(index: pd.DatetimeIndex) -> str:
    """Finest rule that still yields a readable number of points.

    Choosing by span alone collapsed a year of daily data into 4 quarters, which
    averages away exactly the spikes anomaly detection exists to find. Resolution
    is preserved until the series would get unwieldy to plot.
    """
    span_days = max((index.max() - index.min()).days, 1)
    for rule, days_per_point in (("D", 1), ("W", 7), ("ME", 30), ("QE", 91), ("YE", 365)):
        if span_days / days_per_point <= MAX_SERIES_POINTS:
            return rule
    return "YE"


# ---------------------------------------------------------------------------
# time series
# ---------------------------------------------------------------------------

def time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str | None = None,
    agg: str = "sum",
    freq: str | None = None,
) -> dict[str, Any]:
    """Resample to a timeline and describe how it moves."""
    dates = as_datetime(df, date_column)
    frame = pd.DataFrame({"_when": dates})

    if value_column:
        if value_column not in df.columns:
            raise ValidationError(f"Unknown column '{value_column}'")
        if agg in {"sum", "mean", "median", "std"} and not pd.api.types.is_numeric_dtype(df[value_column]):
            raise ValidationError(f"Cannot compute {agg} of non-numeric column '{value_column}'")
        frame["_value"] = df[value_column].to_numpy()
    else:
        frame["_value"] = 1.0
        agg = "sum"

    frame = frame.dropna(subset=["_when"]).set_index("_when").sort_index()
    if len(frame) < MIN_SERIES_POINTS:
        raise ValidationError("Not enough dated rows to build a timeline.")

    rule = freq or _auto_freq(frame.index)
    series = frame["_value"].resample(rule).agg(agg).dropna()
    if len(series) < 2:
        raise ValidationError("The timeline collapses to a single period — try a finer frequency.")

    values = series.to_numpy(dtype=float)
    first, last = float(values[0]), float(values[-1])

    # Trend: ordinary least squares against period index.
    x = np.arange(len(values), dtype=float)
    slope, intercept, r_value, p_value, _ = stats.linregress(x, values)
    direction = (
        "flat" if p_value > ALPHA else "rising" if slope > 0 else "falling"
    )

    changes = pd.Series(values).pct_change().replace([np.inf, -np.inf], np.nan) * 100
    window = max(2, min(7, len(values) // 3))
    rolling = pd.Series(values).rolling(window, min_periods=1).mean()

    # Anomalies on the residual so a strong trend does not flag every late point.
    residual = values - (slope * x + intercept)
    spread = float(np.std(residual))
    anomalies = []
    if spread > 0:
        for i, r in enumerate(residual):
            z = float(r / spread)
            if abs(z) >= 3:
                anomalies.append(
                    {"period": str(series.index[i].date()), "value": to_jsonable(values[i]),
                     "z_score": round(z, 2), "direction": "spike" if z > 0 else "dip"}
                )

    return {
        "date_column": str(date_column),
        "value_column": str(value_column) if value_column else None,
        "agg": agg,
        "freq": rule,
        "points": [
            {"period": str(idx.date()), "value": to_jsonable(v)}
            for idx, v in zip(series.index, values)
        ],
        "moving_average": [to_jsonable(v) for v in rolling.to_numpy()],
        "period_change_pct": [None if pd.isna(c) else round(float(c), 2) for c in changes],
        "trend": {
            "direction": direction,
            "slope_per_period": to_jsonable(slope),
            "r_squared": round(float(r_value**2), 4),
            "p_value": to_jsonable(p_value),
            "significant": bool(p_value <= ALPHA),
        },
        "total_change_pct": None if first == 0 else round((last - first) / abs(first) * 100, 2),
        "first": {"period": str(series.index[0].date()), "value": to_jsonable(first)},
        "last": {"period": str(series.index[-1].date()), "value": to_jsonable(last)},
        "peak": {"period": str(series.idxmax().date()), "value": to_jsonable(series.max())},
        "trough": {"period": str(series.idxmin().date()), "value": to_jsonable(series.min())},
        "anomalies": anomalies,
        "seasonality": _seasonality(frame["_value"], frame.index, agg),
    }


def _seasonality(values: pd.Series, index: pd.DatetimeIndex, agg: str) -> dict[str, Any] | None:
    """Average level by month and weekday, when the span justifies it."""
    span_days = (index.max() - index.min()).days
    out: dict[str, Any] = {}
    if span_days >= 365:
        by_month = values.groupby(index.month).agg(agg)
        if len(by_month) >= 3:
            names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            out["by_month"] = [
                {"label": names[int(m) - 1], "value": to_jsonable(v)} for m, v in by_month.items()
            ]
    if span_days >= 21:
        by_weekday = values.groupby(index.dayofweek).agg(agg)
        if len(by_weekday) >= 3:
            names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            out["by_weekday"] = [
                {"label": names[int(d)], "value": to_jsonable(v)} for d, v in by_weekday.items()
            ]
    return out or None


# ---------------------------------------------------------------------------
# segment comparison
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)


def compare_segments(df: pd.DataFrame, dimension: str, metric: str | None = None) -> dict[str, Any]:
    """Compare a metric across groups and say whether the gap is real.

    Numeric metric: Welch t-test (2 groups) or one-way ANOVA (3+), each paired
    with a rank-based test that does not assume normality. Without a numeric
    metric it becomes a chi-square test of independence on the counts.
    """
    if dimension not in df.columns:
        raise ValidationError(f"Unknown column '{dimension}'")
    if metric is not None and metric not in df.columns:
        raise ValidationError(f"Unknown column '{metric}'")

    groups = df[dimension].dropna()
    if groups.nunique() < 2:
        raise ValidationError(f"'{dimension}' has fewer than two groups to compare.")
    if groups.nunique() > MAX_SEGMENTS:
        raise ValidationError(
            f"'{dimension}' has {groups.nunique()} groups — too many to compare. "
            "Pick a column with fewer distinct values."
        )

    numeric_metric = metric is not None and pd.api.types.is_numeric_dtype(df[metric])

    if numeric_metric:
        buckets, segments = [], []
        for name, chunk in df.dropna(subset=[dimension, metric]).groupby(dimension, observed=True):
            values = chunk[metric].to_numpy(dtype=float)
            if len(values) < 2:
                continue
            buckets.append(values)
            segments.append({
                "group": str(name),
                "count": int(len(values)),
                "mean": to_jsonable(values.mean()),
                "median": to_jsonable(np.median(values)),
                "std": to_jsonable(values.std(ddof=1)),
                "min": to_jsonable(values.min()),
                "max": to_jsonable(values.max()),
            })
        if len(buckets) < 2:
            raise ValidationError("Not enough data per group to compare.")

        segments.sort(key=lambda s: (s["mean"] is None, -(s["mean"] or 0)))
        if len(buckets) == 2:
            stat, p = stats.ttest_ind(buckets[0], buckets[1], equal_var=False)
            _, p_rank = stats.mannwhitneyu(buckets[0], buckets[1], alternative="two-sided")
            test, effect = "Welch t-test", {"cohens_d": round(_cohens_d(buckets[0], buckets[1]), 4)}
            rank_test = "Mann-Whitney U"
        else:
            stat, p = stats.f_oneway(*buckets)
            _, p_rank = stats.kruskal(*buckets)
            grand = np.concatenate(buckets)
            ss_between = float(sum(len(b) * (b.mean() - grand.mean()) ** 2 for b in buckets))
            ss_total = float(((grand - grand.mean()) ** 2).sum())
            test = "One-way ANOVA"
            effect = {"eta_squared": round(ss_between / ss_total, 4) if ss_total else 0.0}
            rank_test = "Kruskal-Wallis"

        best, worst = segments[0], segments[-1]
        gap = (best["mean"] or 0) - (worst["mean"] or 0)
        return {
            "dimension": str(dimension), "metric": str(metric), "kind": "numeric",
            "segments": segments,
            "test": {
                "name": test, "statistic": to_jsonable(stat), "p_value": to_jsonable(p),
                "significant": bool(p <= ALPHA), "alpha": ALPHA,
                "robust_check": {"name": rank_test, "p_value": to_jsonable(p_rank),
                                 "agrees": bool((p <= ALPHA) == (p_rank <= ALPHA))},
                "effect_size": effect,
            },
            "gap": {"best": best["group"], "worst": worst["group"], "difference": to_jsonable(gap)},
            "verdict": (
                f"The difference in {metric} across {dimension} is statistically significant "
                f"(p={p:.4g})." if p <= ALPHA else
                f"The differences in {metric} across {dimension} are within noise (p={p:.4g})."
            ),
        }

    # Categorical (or absent) metric -> independence test on counts.
    if metric is None:
        counts = df[dimension].value_counts(dropna=True)
        expected = len(df.dropna(subset=[dimension])) / len(counts)
        stat, p = stats.chisquare(counts.to_numpy())
        return {
            "dimension": str(dimension), "metric": None, "kind": "distribution",
            "segments": [
                {"group": str(k), "count": int(v), "share_pct": round(100 * v / counts.sum(), 2)}
                for k, v in counts.items()
            ],
            "test": {"name": "Chi-square goodness of fit", "statistic": to_jsonable(stat),
                     "p_value": to_jsonable(p), "significant": bool(p <= ALPHA), "alpha": ALPHA,
                     "expected_per_group": round(expected, 2)},
            "verdict": (
                f"'{dimension}' is unevenly distributed (p={p:.4g})." if p <= ALPHA
                else f"'{dimension}' is close to evenly distributed (p={p:.4g})."
            ),
        }

    table = pd.crosstab(df[dimension], df[metric])
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        raise ValidationError(f"Not enough overlap between '{dimension}' and '{metric}' to test.")
    stat, p, dof, _ = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    min_dim = min(table.shape) - 1
    cramers_v = float(np.sqrt((stat / n) / min_dim)) if n and min_dim else 0.0
    return {
        "dimension": str(dimension), "metric": str(metric), "kind": "categorical",
        "levels": [str(c) for c in table.columns],
        "segments": [
            {"group": str(idx), "counts": [int(v) for v in row], "total": int(row.sum())}
            for idx, row in zip(table.index, table.to_numpy())
        ],
        "test": {"name": "Chi-square test of independence", "statistic": to_jsonable(stat),
                 "p_value": to_jsonable(p), "dof": int(dof), "significant": bool(p <= ALPHA),
                 "alpha": ALPHA, "effect_size": {"cramers_v": round(cramers_v, 4)}},
        "verdict": (
            f"'{metric}' depends on '{dimension}' (p={p:.4g})." if p <= ALPHA
            else f"'{metric}' looks independent of '{dimension}' (p={p:.4g})."
        ),
    }
