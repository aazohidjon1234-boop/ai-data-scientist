"""Derive new columns from existing ones — products, ratios, differences.

Interactions are the one kind of signal a model cannot always find on its own:
a linear model never sees `age × max_heart_rate` unless you build it, and trees
approximate it only with many splits.

The risk is the opposite of the reward. Every pair doubles the search space, so
candidates are generated only among the columns that already matter, ranked by
mutual information, and kept only if the whole set beats the plain one under the
same cross-validation as every other recipe. A derived column that merely looks
clever is discarded like any other failed idea.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

MAX_PARENTS = 6           # pairs grow quadratically; 6 parents = 15 pairs
MAX_CANDIDATES = 8        # how many derived columns to offer at most
MIN_ROWS = 40
EPSILON = 1e-9

Kind = Literal["product", "ratio", "difference"]


def _safe_name(kind: Kind, a: str, b: str) -> str:
    symbol = {"product": "x", "ratio": "per", "difference": "minus"}[kind]
    return f"{a}_{symbol}_{b}"


def compute(df: pd.DataFrame, spec: dict[str, Any]) -> pd.Series | None:
    """Build one derived column from its spec. None when it is not computable."""
    a, b, kind = spec.get("a"), spec.get("b"), spec.get("kind")
    if a not in df.columns or b not in df.columns:
        return None
    left, right = pd.to_numeric(df[a], errors="coerce"), pd.to_numeric(df[b], errors="coerce")
    if kind == "product":
        out = left * right
    elif kind == "difference":
        out = left - right
    elif kind == "ratio":
        # Guard the denominator rather than dropping the column: a zero here is
        # ordinary data, not an error.
        out = left / right.where(right.abs() > EPSILON, np.nan)
    else:
        return None
    # A division can still overflow to inf even with the guard above.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out if out.notna().any() and out.nunique(dropna=True) > 1 else None


def build_features(df: pd.DataFrame, specs: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """Add every computable derived column. Returns the frame and their names."""
    if not specs:
        return df, []
    out = df.copy()
    added: list[str] = []
    for spec in specs:
        name = spec.get("name") or _safe_name(spec["kind"], spec["a"], spec["b"])
        if name in out.columns:
            continue
        series = compute(out, spec)
        if series is None:
            continue
        out[name] = series
        added.append(name)
    return out, added


def _relevance(frame: pd.DataFrame, columns: list[str], target: str,
               problem_type: str) -> dict[str, float]:
    data = frame[columns + [target]].dropna(subset=[target])
    if len(data) < MIN_ROWS or not columns:
        return dict.fromkeys(columns, 0.0)
    X = data[columns].apply(lambda s: s.fillna(s.median()))
    y = data[target]
    try:
        if problem_type == "classification":
            scores = mutual_info_classif(X, y.astype(str), random_state=0)
        else:
            numeric_y = pd.to_numeric(y, errors="coerce")
            keep = numeric_y.notna()
            if keep.sum() < MIN_ROWS:
                return dict.fromkeys(columns, 0.0)
            scores = mutual_info_regression(X[keep], numeric_y[keep], random_state=0)
    except Exception:
        return dict.fromkeys(columns, 0.0)
    return {c: float(s) for c, s in zip(columns, scores)}


def propose(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    base_features: list[str],
    limit: int = MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """Rank derived columns and return the ones worth trying.

    A candidate has to beat *both* of its parents on mutual information —
    otherwise it repeats what the model can already see and only adds noise.
    """
    numeric = [
        c for c in base_features
        if c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and df[c].nunique(dropna=True) > 2
    ]
    if len(numeric) < 2 or len(df) < MIN_ROWS:
        return []

    parent_scores = _relevance(df, numeric, target, problem_type)
    parents = sorted(numeric, key=lambda c: -parent_scores.get(c, 0.0))[:MAX_PARENTS]

    candidates: list[dict[str, Any]] = []
    working = df.copy()
    for i, a in enumerate(parents):
        for b in parents[i + 1:]:
            for kind in ("product", "ratio", "difference"):
                spec = {"kind": kind, "a": a, "b": b, "name": _safe_name(kind, a, b)}
                series = compute(working, spec)
                if series is None:
                    continue
                working[spec["name"]] = series
                candidates.append(spec)
    if not candidates:
        return []

    names = [c["name"] for c in candidates]
    scores = _relevance(working, names, target, problem_type)
    ranked = []
    for spec in candidates:
        score = scores.get(spec["name"], 0.0)
        best_parent = max(parent_scores.get(spec["a"], 0.0), parent_scores.get(spec["b"], 0.0))
        if score <= best_parent:
            continue  # adds nothing its parents did not already carry
        ranked.append({**spec, "relevance": round(score, 4),
                       "parent_relevance": round(best_parent, 4)})
    ranked.sort(key=lambda s: -s["relevance"])
    return ranked[:limit]
