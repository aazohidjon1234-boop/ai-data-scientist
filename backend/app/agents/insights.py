"""Auto-insights — what an analyst notices before being asked.

Scans a dataset and returns a ranked list of findings: strong relationships,
lopsided categories, trends, anomalies, outliers and quality risks. Each finding
carries the numbers it was derived from, so nothing here is an opinion.

Ranking is by `importance` (0-100), computed from effect size — correlation
strength, dominance share, trend R², significance, outlier share — not from
wording. That keeps "80% of rows are one category" above "3 rows are missing".
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..tools import data_tools as dt
from ..tools.analytics_tools import ALPHA, detect_datetime_columns, time_series
from ..utils.jsonutils import to_jsonable

STRONG_CORR = 0.5
DOMINANT_SHARE = 65.0
HIGH_MISSING_PCT = 20.0
OUTLIER_PCT = 5.0
SKEW_LIMIT = 1.5
MAX_DIMENSION_CARDINALITY = 15
MIN_GROUP_ROWS = 8


def _insight(kind: str, title: str, detail: str, importance: float,
             columns: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "title": title,
        "detail": detail,
        "importance": round(max(0.0, min(100.0, float(importance))), 1),
        "columns": [str(c) for c in columns],
        "evidence": to_jsonable(evidence),
    }


def _dimensions(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if (not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
            or df[c].nunique(dropna=True) <= MAX_DIMENSION_CARDINALITY)
        and 2 <= df[c].nunique(dropna=True) <= MAX_DIMENSION_CARDINALITY
    ]


def _measures(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and df[c].nunique(dropna=True) > MAX_DIMENSION_CARDINALITY
    ]


# ---------------------------------------------------------------------------
# individual scanners
# ---------------------------------------------------------------------------

def _correlation_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    corr = dt.generate_correlation_matrix(df)
    out = []
    for pair in (corr or {}).get("top_pairs", [])[:5]:
        r = float(pair.get("corr") or 0)
        if abs(r) < STRONG_CORR:
            continue
        direction = "rises with" if r > 0 else "falls as"
        out.append(_insight(
            "correlation",
            f"{pair['a']} {direction} {pair['b']}",
            f"Pearson correlation is {r:+.2f}. Strong pairs often mean one column "
            "duplicates information in the other, which matters when modelling.",
            abs(r) * 100,
            [pair["a"], pair["b"]],
            {"correlation": r},
        ))
    return out


def _distribution_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for column in _dimensions(df):
        counts = df[column].value_counts(dropna=True)
        if counts.empty:
            continue
        share = 100.0 * counts.iloc[0] / counts.sum()
        if share >= DOMINANT_SHARE:
            out.append(_insight(
                "imbalance",
                f"'{column}' is dominated by {counts.index[0]}",
                f"{share:.1f}% of rows fall in a single category ({counts.iloc[0]:,} of "
                f"{counts.sum():,}). If this is the target, accuracy will look high for "
                "the wrong reason — check per-class recall instead.",
                share,
                [column],
                {"top_value": str(counts.index[0]), "share_pct": round(share, 2),
                 "count": int(counts.iloc[0]), "total": int(counts.sum())},
            ))
    return out


def _skew_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for column in _measures(df):
        values = df[column].dropna()
        if len(values) < 10:
            continue
        skew = float(stats.skew(values.to_numpy(dtype=float)))
        if abs(skew) >= SKEW_LIMIT:
            side = "right" if skew > 0 else "left"
            out.append(_insight(
                "skew",
                f"'{column}' is heavily {side}-skewed",
                f"Skewness is {skew:+.2f}, so the mean ({values.mean():,.2f}) is pulled away "
                f"from the median ({values.median():,.2f}). Prefer the median when summarising.",
                min(abs(skew) * 25, 80),
                [column],
                {"skew": round(skew, 3), "mean": to_jsonable(values.mean()),
                 "median": to_jsonable(values.median())},
            ))
    return out


def _quality_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    missing = dt.detect_missing_values(df)
    for entry in missing.get("per_column", []):
        if entry["pct"] >= HIGH_MISSING_PCT:
            out.append(_insight(
                "data_quality",
                f"'{entry['column']}' is {entry['pct']:.1f}% empty",
                f"{entry['missing']:,} rows have no value. Imputation will invent "
                "a large share of this column — treat conclusions about it carefully.",
                min(entry["pct"] * 1.5, 95),
                [entry["column"]],
                {"missing": entry["missing"], "pct": entry["pct"]},
            ))

    profile = dt.analyze_dataset(df)
    duplicates = int(profile.get("duplicate_rows") or 0)
    if duplicates:
        pct = 100.0 * duplicates / max(len(df), 1)
        out.append(_insight(
            "data_quality",
            f"{duplicates:,} duplicate rows",
            f"{pct:.1f}% of rows are exact copies. They inflate counts and can leak "
            "between train and test splits.",
            min(pct * 3, 90),
            [],
            {"duplicate_rows": duplicates, "pct": round(pct, 2)},
        ))

    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant:
        out.append(_insight(
            "data_quality",
            f"{len(constant)} column(s) never change",
            "Constant columns carry no information and are dropped before modelling: "
            + ", ".join(map(str, constant[:6])),
            40,
            constant[:6],
            {"columns": [str(c) for c in constant]},
        ))

    outliers = dt.detect_outliers(df)
    for column, info in (outliers.get("columns") or {}).items():
        count = int(info.get("count", 0)) if isinstance(info, dict) else int(info)
        pct = 100.0 * count / max(len(df), 1)
        if pct >= OUTLIER_PCT:
            out.append(_insight(
                "outliers",
                f"'{column}' has {count:,} outliers ({pct:.1f}%)",
                "Values beyond 1.5x the interquartile range. Genuine extremes or data "
                "entry errors — worth checking before averaging this column.",
                min(pct * 4, 85),
                [column],
                {"outlier_count": count, "pct": round(pct, 2)},
            ))
    return out


def _segment_insights(df: pd.DataFrame, max_tests: int = 12) -> list[dict[str, Any]]:
    """Numeric measures that genuinely differ across a low-cardinality dimension."""
    out: list[dict[str, Any]] = []
    tests = 0
    for dimension in _dimensions(df)[:5]:
        for measure in _measures(df)[:5]:
            if tests >= max_tests:
                return out
            frame = df[[dimension, measure]].dropna()
            buckets = [
                g[measure].to_numpy(dtype=float)
                for _, g in frame.groupby(dimension, observed=True)
                if len(g) >= MIN_GROUP_ROWS
            ]
            if len(buckets) < 2:
                continue
            tests += 1
            try:
                stat, p = stats.f_oneway(*buckets)
            except Exception:
                continue
            if not np.isfinite(p) or p > ALPHA:
                continue
            grand = np.concatenate(buckets)
            ss_total = float(((grand - grand.mean()) ** 2).sum())
            ss_between = float(sum(len(b) * (b.mean() - grand.mean()) ** 2 for b in buckets))
            eta = ss_between / ss_total if ss_total else 0.0
            if eta < 0.02:  # statistically real but practically negligible
                continue
            means = frame.groupby(dimension, observed=True)[measure].mean().sort_values()
            out.append(_insight(
                "segment_difference",
                f"'{measure}' differs by '{dimension}'",
                f"{means.index[-1]} averages {means.iloc[-1]:,.2f} versus {means.iloc[0]:,.2f} "
                f"for {means.index[0]} (p={p:.3g}, eta²={eta:.2f}). The gap is larger than "
                "sampling noise.",
                min(eta * 200, 95),
                [dimension, measure],
                {"p_value": to_jsonable(p), "eta_squared": round(eta, 4),
                 "highest": {"group": str(means.index[-1]), "mean": to_jsonable(means.iloc[-1])},
                 "lowest": {"group": str(means.index[0]), "mean": to_jsonable(means.iloc[0])}},
            ))
    return out


def _trend_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    date_columns = detect_datetime_columns(df)
    if not date_columns:
        return out
    date_column = date_columns[0]
    for measure in _measures(df)[:3]:
        try:
            series = time_series(df, date_column, measure, agg="mean")
        except Exception:
            continue
        trend = series["trend"]
        if trend["direction"] != "flat" and trend["significant"]:
            change = series.get("total_change_pct")
            out.append(_insight(
                "trend",
                f"'{measure}' is {trend['direction']} over time",
                f"Between {series['first']['period']} and {series['last']['period']} it moved "
                + (f"{change:+.1f}% " if change is not None else "")
                + f"(R²={trend['r_squared']:.2f}, p={trend['p_value']:.3g}).",
                min(trend["r_squared"] * 100, 95),
                [date_column, measure],
                {"direction": trend["direction"], "r_squared": trend["r_squared"],
                 "p_value": trend["p_value"], "total_change_pct": change},
            ))
        for anomaly in series.get("anomalies", [])[:2]:
            out.append(_insight(
                "anomaly",
                f"Unusual {anomaly['direction']} in '{measure}' at {anomaly['period']}",
                f"Value {anomaly['value']:,.2f} sits {abs(anomaly['z_score']):.1f} standard "
                "deviations off the trend line.",
                min(abs(anomaly["z_score"]) * 20, 90),
                [date_column, measure],
                anomaly,
            ))
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def generate_insights(df: pd.DataFrame, limit: int = 12) -> dict[str, Any]:
    """Rank everything noteworthy about a dataset."""
    findings: list[dict[str, Any]] = []
    for scanner in (
        _quality_insights,
        _correlation_insights,
        _distribution_insights,
        _segment_insights,
        _trend_insights,
        _skew_insights,
    ):
        try:
            findings.extend(scanner(df))
        except Exception:
            # One failing scanner must not cost the user every other insight.
            continue

    findings.sort(key=lambda f: f["importance"], reverse=True)
    top = findings[:limit]
    return {
        "count": len(findings),
        "returned": len(top),
        "insights": top,
        "by_kind": {
            kind: sum(1 for f in findings if f["kind"] == kind)
            for kind in sorted({f["kind"] for f in findings})
        },
        "headline": top[0]["title"] if top else "Nothing unusual stood out in this dataset.",
        "tools_used": ["analyze_dataset", "detect_missing_values", "detect_outliers",
                       "generate_correlation_matrix", "compare_segments", "time_series"],
    }
