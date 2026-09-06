"""Try concrete ways to improve a model — and report what actually worked.

"Remove the outliers and it will be more accurate" is a guess. This module
turns each such idea into a **recipe**, cross-validates it against the untouched
baseline, and returns the measured scores so the user can decide. A recipe that
makes things worse is reported as worse; nothing is applied automatically.

Each recipe is scored the way the real pipeline picks a winner — several fast,
diverse models, best score wins — so an improvement here is likely to survive
the full model zoo rather than being an artefact of one estimator.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, SVR

from ..core.config import get_settings
from ..exceptions import ValidationError
from ..tools.data_tools import prepare_features
from ..utils.jsonutils import to_jsonable
from .feature_advisor import suggest_features
from .tuner import is_tunable, tune_model

EVAL_ROWS = 4_000
EVAL_FOLDS = 3
IQR_MULTIPLIER = 1.5
MIN_UNIQUE_FOR_OUTLIERS = 8
MAX_ROW_LOSS = 0.25       # refuse to throw away more than a quarter of the data
MEANINGFUL_GAIN = 0.005


def outlier_mask(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Rows that sit outside 1.5·IQR on any numeric column considered."""
    mask = pd.Series(False, index=df.index)
    for column in columns:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        clean = series.dropna()
        if clean.nunique() < MIN_UNIQUE_FOR_OUTLIERS:
            continue  # discrete codes, not measurements
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low, high = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        mask |= (series < low) | (series > high)
    return mask.fillna(False)


def _zoo(problem_type: str) -> list[tuple[str, Any]]:
    if problem_type == "classification":
        return [
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Random Forest", RandomForestClassifier(n_estimators=80, random_state=0, n_jobs=-1)),
            ("SVM", SVC()),
        ]
    return [
        ("Ridge", Ridge()),
        ("Random Forest", RandomForestRegressor(n_estimators=80, random_state=0, n_jobs=-1)),
        ("SVR", SVR()),
    ]


def _score(df: pd.DataFrame, target: str, problem_type: str,
           features: list[str] | None) -> tuple[float | None, str | None]:
    """Best cross-validated score across a few fast models, like the real run."""
    if df.empty:
        return None, None
    frame = df
    if len(frame) > EVAL_ROWS:
        frame = frame.sample(n=EVAL_ROWS, random_state=get_settings().random_state)
    try:
        X, y, _ = prepare_features(frame, target, problem_type, features)
    except ValidationError:
        return None, None
    if y is None or X.shape[1] == 0 or len(X) < EVAL_FOLDS * 3:
        return None, None
    scoring = "f1_macro" if problem_type == "classification" else "r2"
    if problem_type == "classification" and pd.Series(y).value_counts().min() < EVAL_FOLDS:
        return None, None

    best, winner = None, None
    for name, model in _zoo(problem_type):
        try:
            score = float(np.mean(cross_val_score(model, X, y, cv=EVAL_FOLDS,
                                                  scoring=scoring, n_jobs=-1)))
        except Exception:
            continue
        if best is None or score > best:
            best, winner = score, name
    return best, winner


def suggest_improvements(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    current_features: list[str] | None = None,
    current_drops_outliers: bool = False,
    current_best_model: str | None = None,
) -> dict[str, Any]:
    """Compare options against the run the user actually has.

    Measuring against the untouched data would keep returning the same table
    after a recipe was applied, so "test again" looked frozen and an already
    applied change was offered forever.
    """
    if target not in df.columns:
        raise ValidationError(f"Target column '{target}' is not in this dataset.")
    if problem_type == "clustering":
        raise ValidationError("Improvements are measured against a target, so clustering is out of scope.")

    work = df.dropna(subset=[target])
    advice = suggest_features(work, target, problem_type)
    selected = advice["recommended"]
    usable = advice["all_usable"]

    numeric_inputs = [c for c in usable if pd.api.types.is_numeric_dtype(work[c])]
    mask = outlier_mask(work, numeric_inputs)
    removed = int(mask.sum())
    loss = removed / max(len(work), 1)
    trimmed = work[~mask] if 0 < loss <= MAX_ROW_LOSS else None

    metric = "macro F1" if problem_type == "classification" else "R²"
    recipes: list[dict[str, Any]] = []

    def add(key: str, label: str, why: str, frame: pd.DataFrame,
            features: list[str] | None, changes: dict[str, Any]) -> None:
        score, winner = _score(frame, target, problem_type, features)
        recipes.append({
            "key": key, "label": label, "why": why, "score": score,
            "best_model": winner, "rows_used": int(len(frame)),
            "features_used": len(features) if features else len(usable),
            "changes": changes,
        })

    # The baseline is whatever is in use right now, not the untouched file.
    baseline_features = [c for c in (current_features or usable) if c in work.columns] or usable
    baseline_frame = trimmed if (current_drops_outliers and trimmed is not None) else work
    baseline_note = []
    if current_features:
        baseline_note.append(f"{len(baseline_features)} chosen column(s)")
    if current_drops_outliers:
        baseline_note.append("outlier rows already removed")
    add("baseline", "Keep the current setup",
        ("Your latest run: " + ", ".join(baseline_note) + ".") if baseline_note
        else "Everything usable, every row — the run you already have.",
        baseline_frame, baseline_features,
        {"features": current_features, "drop_outliers": current_drops_outliers})

    same_features = sorted(selected) == sorted(baseline_features)
    if selected and not same_features:
        add("features", "Train on the informative columns only",
            f"Uses {len(selected)} of {len(usable)} usable column(s) — drops the ones "
            "carrying little signal.",
            work, selected, {"features": selected, "drop_outliers": False})

    if trimmed is not None and not current_drops_outliers:
        add("outliers", "Remove outlier rows",
            f"Drops {removed} row(s) ({loss * 100:.1f}%) outside 1.5×IQR on a numeric column.",
            trimmed, baseline_features, {"features": current_features, "drop_outliers": True})

        if selected and not same_features:
            add("both", "Informative columns and no outliers",
                "Both changes together.", trimmed, selected,
                {"features": selected, "drop_outliers": True})

    # Tuning the winner, measured on the same folds as everything else.
    tuning: dict[str, Any] | None = None
    if current_best_model and is_tunable(current_best_model):
        try:
            frame = baseline_frame
            if len(frame) > EVAL_ROWS:
                frame = frame.sample(n=EVAL_ROWS, random_state=get_settings().random_state)
            X, y, _ = prepare_features(frame, target, problem_type, baseline_features)
            tuning = tune_model(X, y, problem_type, current_best_model)
        except Exception:
            tuning = None
    if tuning:
        recipes.append({
            "key": "tuned",
            "label": f"Tune {current_best_model}",
            "why": (
                f"Randomised search over {tuning['iterations']} parameter combinations "
                f"for the model that currently wins."
            ),
            "score": tuning["cv_score"],
            "best_model": current_best_model,
            "rows_used": int(len(baseline_frame)),
            "features_used": len(baseline_features),
            "changes": {
                "features": current_features,
                "drop_outliers": current_drops_outliers,
                "tune": True,
            },
            "params": tuning["params"],
        })

    scored = [r for r in recipes if r["score"] is not None]
    baseline = next((r for r in recipes if r["key"] == "baseline"), None)
    base_score = baseline["score"] if baseline else None
    for recipe in recipes:
        if recipe["score"] is not None and base_score is not None:
            recipe["delta"] = round(recipe["score"] - base_score, 4)
        else:
            recipe["delta"] = None
    scored.sort(key=lambda r: -r["score"])
    best = scored[0] if scored else None

    if best is None or base_score is None:
        summary = "This dataset could not be scored reliably, so there is nothing to compare."
    elif best["key"] == "baseline" or (best["score"] - base_score) < MEANINGFUL_GAIN:
        summary = (
            f"Nothing beat your current setup by a meaningful margin — it scores {metric} "
            f"{base_score:.4f}. The data is already in reasonable shape; more rows or new "
            "features would help more than further cleaning."
        )
    else:
        summary = (
            f"Best option: **{best['label']}** — cross-validated {metric} "
            f"{base_score:.4f} → {best['score']:.4f} (+{best['score'] - base_score:.4f})."
        )

    return to_jsonable({
        "target": target,
        "problem_type": problem_type,
        "metric": metric,
        "folds": EVAL_FOLDS,
        "baseline_score": base_score,
        "recommended": best["key"] if best and best["key"] != "baseline"
                       and base_score is not None
                       and (best["score"] - base_score) >= MEANINGFUL_GAIN else "baseline",
        "recipes": recipes,
        "tuning": tuning,
        "outliers": {
            "rows_flagged": removed,
            "pct": round(loss * 100, 2),
            "columns_checked": numeric_inputs,
            "skipped_reason": (
                f"would remove {loss * 100:.0f}% of the data" if loss > MAX_ROW_LOSS else None
            ),
        },
        "summary": summary,
        "tools_used": ["detect_outliers", "mutual_information", "cross_val_score",
                       "randomized_search"],
    })
