"""Recommend which columns to train on — and measure whether it helps.

Two halves:

* **Screening.** Every column gets a usability verdict (constant, identifier,
  mostly empty, too many categories) and, when it survives, a relevance score
  against the target from mutual information — which catches non-linear
  relationships that a correlation coefficient misses.
* **Evidence.** The recommendation is then *measured*: the same quick model is
  cross-validated on all usable columns and on the recommendation, and both
  scores are returned. Saying "this improves accuracy" without checking would
  be a guess, and the whole point of this project is that numbers come from
  Python rather than from a sentence.

If the trimmed set scores worse, that is reported plainly instead of hidden.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score

from ..core.config import get_settings
from ..exceptions import ValidationError
from ..tools.data_tools import coerce_numeric_columns, prepare_features
from ..utils.jsonutils import to_jsonable

# A column whose values are nearly all distinct carries no pattern to learn —
# it identifies rows instead of describing them.
IDENTIFIER_UNIQUE_RATIO = 0.9
MOSTLY_EMPTY_PCT = 50.0
RELEVANCE_FLOOR = 0.01          # mutual information below this is noise
EVAL_ROWS = 5_000               # cap for the cross-validated check
EVAL_FOLDS = 3
LEAKAGE_CORR = 0.98


def _usability(df: pd.DataFrame, column: str, rows: int) -> tuple[bool, str]:
    """Can this column be used at all? Returns (usable, reason)."""
    settings = get_settings()
    series = df[column]
    missing_pct = 100.0 * series.isna().sum() / max(rows, 1)
    unique = series.nunique(dropna=True)

    if unique <= 1:
        return False, "every row has the same value"
    if missing_pct >= MOSTLY_EMPTY_PCT:
        return False, f"{missing_pct:.0f}% of values are missing"
    is_categorical = not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
    if is_categorical and unique > settings.max_cardinality:
        return False, f"{unique} distinct categories — too many to encode"

    ratio = unique / max(rows, 1)
    if is_categorical and ratio >= IDENTIFIER_UNIQUE_RATIO:
        return False, "looks like a row identifier, not a measurement"
    if not is_categorical and _is_sequential_id(series, unique, ratio):
        return False, "looks like a running row number, not a measurement"
    return True, ""


def _is_sequential_id(series: pd.Series, unique: int, ratio: float) -> bool:
    """Integer column that just counts rows (1, 2, 3, …), duplicates allowed.

    Requiring `unique == len(df)` misses the common case where the file has a
    few duplicated rows — Titanic's PassengerId is 891 unique across 896 rows
    and slipped through as a real feature.
    """
    if ratio < 0.95 or unique < 20:
        return False
    values = series.dropna()
    if values.empty or not np.all(np.equal(np.mod(values.to_numpy(dtype=float), 1), 0)):
        return False  # not integer-valued
    span = float(values.max() - values.min()) + 1
    # A counter covers its range densely; a genuine measurement rarely does.
    return bool(span <= unique * 1.5)


def _relevance(df: pd.DataFrame, columns: list[str], target: str, problem_type: str) -> dict[str, float]:
    """Mutual information between each candidate column and the target."""
    if not columns:
        return {}
    frame = df[columns + [target]].dropna(subset=[target])
    if len(frame) < 10:
        return dict.fromkeys(columns, 0.0)

    encoded = pd.DataFrame(index=frame.index)
    # Mutual information estimates discrete and continuous features differently.
    # Treating an integer code like Pclass as continuous makes the k-NN estimator
    # return 0 for a column that genuinely predicts the target.
    discrete: list[bool] = []
    for column in columns:
        series = frame[column]
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if numeric:
            filled = series.fillna(series.median())
            encoded[column] = filled
            integral = bool(np.all(np.equal(np.mod(filled.to_numpy(dtype=float), 1), 0)))
            discrete.append(integral and filled.nunique() <= 20)
        else:
            encoded[column] = series.astype(str).fillna("missing").astype("category").cat.codes
            discrete.append(True)

    y = frame[target]
    mask = np.array(discrete, dtype=bool)
    try:
        if problem_type == "classification":
            scores = mutual_info_classif(
                encoded, y.astype(str), discrete_features=mask, random_state=0
            )
        else:
            y_num = pd.to_numeric(y, errors="coerce")
            keep = y_num.notna()
            if keep.sum() < 10:
                return dict.fromkeys(columns, 0.0)
            scores = mutual_info_regression(
                encoded[keep], y_num[keep], discrete_features=mask, random_state=0
            )
    except Exception:
        return dict.fromkeys(columns, 0.0)
    return {c: float(s) for c, s in zip(columns, scores)}


def _leaky(df: pd.DataFrame, column: str, target: str) -> bool:
    """A feature almost perfectly matching the target is usually leakage."""
    if not (pd.api.types.is_numeric_dtype(df[column]) and pd.api.types.is_numeric_dtype(df[target])):
        return False
    pair = df[[column, target]].dropna()
    if len(pair) < 10 or pair[column].nunique() <= 1:
        return False
    corr = pair[column].corr(pair[target])
    return bool(pd.notna(corr) and abs(corr) >= LEAKAGE_CORR)


def _cv_score(df: pd.DataFrame, target: str, problem_type: str, features: list[str]) -> float | None:
    """Cross-validated score for one feature set, using a single quick model."""
    if not features:
        return None
    frame = df
    if len(frame) > EVAL_ROWS:
        frame = frame.sample(n=EVAL_ROWS, random_state=get_settings().random_state)
    try:
        X, y, _ = prepare_features(frame, target, problem_type, features)
        if y is None or X.shape[1] == 0:
            return None
        if problem_type == "classification":
            model = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=-1)
            scoring = "f1_macro"
            counts = pd.Series(y).value_counts()
            if counts.min() < EVAL_FOLDS:
                return None  # a class too rare to fold
        else:
            model = RandomForestRegressor(n_estimators=60, random_state=0, n_jobs=-1)
            scoring = "r2"
        scores = cross_val_score(model, X, y, cv=EVAL_FOLDS, scoring=scoring, n_jobs=-1)
        return float(np.mean(scores))
    except Exception:
        return None


def suggest_features(df: pd.DataFrame, target: str, problem_type: str) -> dict[str, Any]:
    if target not in df.columns:
        raise ValidationError(f"Target column '{target}' is not in this dataset.")
    if problem_type == "clustering":
        raise ValidationError(
            "Feature suggestion compares columns against a target, so it needs a supervised task."
        )

    # Read "€110.5M" / "5'7" / "88+2" as numbers first, otherwise they are
    # judged as high-cardinality text and dropped despite carrying real signal.
    df, _coerced = coerce_numeric_columns(df)
    rows = len(df)
    candidates = [c for c in df.columns if c != target]
    screened: list[dict[str, Any]] = []
    columns: list[str] = []
    for column in candidates:
        ok, reason = _usability(df, column, rows)
        if ok:
            columns.append(column)
        screened.append({"column": column, "usable": ok, "reason": reason})

    scores = _relevance(df, columns, target, problem_type)
    ranked = []
    for entry in screened:
        column = entry["column"]
        score = scores.get(column, 0.0)
        leak = entry["usable"] and _leaky(df, column, target)
        if not entry["usable"]:
            verdict, why = "drop", entry["reason"]
        elif leak:
            verdict, why = "review", "almost perfectly matches the target — check for leakage"
        elif score < RELEVANCE_FLOOR:
            verdict, why = "weak", "carries almost no information about the target"
        else:
            verdict, why = "keep", "informative about the target"
        ranked.append({
            "column": column,
            "relevance": round(score, 4),
            "verdict": verdict,
            "reason": why,
            "missing_pct": round(100.0 * df[column].isna().sum() / max(rows, 1), 1),
            "unique": int(df[column].nunique(dropna=True)),
        })
    ranked.sort(key=lambda r: (r["verdict"] != "keep", -r["relevance"]))

    recommended = [r["column"] for r in ranked if r["verdict"] == "keep"]
    all_usable = [r["column"] for r in ranked if r["verdict"] in {"keep", "weak", "review"}]
    if not recommended:
        # Never hand back an empty selection; fall back to everything usable.
        recommended = all_usable

    baseline = _cv_score(df, target, problem_type, all_usable)
    trimmed = _cv_score(df, target, problem_type, recommended) if recommended != all_usable else baseline
    metric = "macro F1" if problem_type == "classification" else "R²"

    if baseline is None or trimmed is None:
        evidence = "The comparison could not be scored on this dataset, so the ranking above is advisory."
    elif recommended == all_usable:
        evidence = (
            f"Every usable column is already worth keeping — cross-validated {metric} "
            f"{baseline:.4f}. The dropped columns were unusable, not merely weak."
        )
    elif trimmed > baseline:
        evidence = (
            f"Measured: cross-validated {metric} improves from {baseline:.4f} to {trimmed:.4f} "
            f"when training on the {len(recommended)} recommended columns."
        )
    elif abs(trimmed - baseline) < 0.005:
        evidence = (
            f"Measured: {metric} is essentially unchanged ({baseline:.4f} vs {trimmed:.4f}), "
            "but the smaller set is faster and easier to explain."
        )
    else:
        evidence = (
            f"Measured: {metric} drops from {baseline:.4f} to {trimmed:.4f} with the trimmed set. "
            "Keeping every usable column looks better for this dataset."
        )

    return to_jsonable({
        "target": target,
        "problem_type": problem_type,
        "columns": ranked,
        "recommended": recommended,
        "all_usable": all_usable,
        "evaluation": {
            "metric": metric,
            "folds": EVAL_FOLDS,
            "model": "Random Forest",
            "score_all_usable": baseline,
            "score_recommended": trimmed,
            "rows_used": min(rows, EVAL_ROWS),
        },
        "summary": evidence,
        "tools_used": ["detect_missing_values", "mutual_information", "cross_val_score"],
    })
