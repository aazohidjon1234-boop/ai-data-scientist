"""Hyperparameter search for the model that already won.

Tuning usually helps, but not always and not for free: a wide grid over 22
models would take far longer than the whole pipeline. So only the winner of the
normal run is tuned, with a randomised search over a small, sensible grid, and
the result is reported next to the untuned score. If the search does not beat
the defaults it says so — a tuned model that scores worse is still worse.

Randomised rather than exhaustive search: with a fixed budget it covers a much
wider range of the space than a grid of the same cost.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV

from ..core.config import get_settings
from ..tools import data_tools

SEARCH_ITERATIONS = 12
SEARCH_FOLDS = 3
MEANINGFUL_GAIN = 0.002

# Grids are deliberately small and centred on the defaults each model already
# uses, so a search is a refinement rather than a lottery.
GRIDS: dict[str, dict[str, Any]] = {
    "Ridge": {"alpha": loguniform(1e-2, 1e2)},
    "Logistic Regression": {"C": loguniform(1e-2, 1e2)},
    "Decision Tree": {
        "max_depth": randint(3, 20),
        "min_samples_leaf": randint(1, 12),
        "min_samples_split": randint(2, 12),
    },
    "Random Forest": {
        "n_estimators": randint(80, 320),
        "max_depth": randint(4, 24),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2", None],
    },
    "Extra Trees": {
        "n_estimators": randint(80, 320),
        "max_depth": randint(4, 24),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2", None],
    },
    "Gradient Boosting": {
        "n_estimators": randint(60, 250),
        "learning_rate": loguniform(0.02, 0.3),
        "max_depth": randint(2, 6),
        "subsample": uniform(0.7, 0.3),
    },
    "Hist Gradient Boosting": {
        "max_iter": randint(100, 400),
        "learning_rate": loguniform(0.02, 0.3),
        "max_leaf_nodes": randint(15, 63),
    },
    "XGBoost": {
        "n_estimators": randint(80, 320),
        "learning_rate": loguniform(0.02, 0.3),
        "max_depth": randint(2, 9),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.6, 0.4),
    },
    "LightGBM": {
        "n_estimators": randint(80, 320),
        "learning_rate": loguniform(0.02, 0.3),
        "num_leaves": randint(15, 63),
        "min_child_samples": randint(5, 40),
    },
    "AdaBoost": {"n_estimators": randint(40, 250), "learning_rate": loguniform(0.05, 1.5)},
    "KNN": {"n_neighbors": randint(3, 30), "weights": ["uniform", "distance"], "p": [1, 2]},
    "SVM": {"C": loguniform(1e-1, 1e2), "gamma": ["scale", "auto"]},
    "SVR": {"C": loguniform(1e-1, 1e2), "gamma": ["scale", "auto"], "epsilon": uniform(0.01, 0.3)},
}


def is_tunable(model_name: str) -> bool:
    return model_name in GRIDS


def tune_model(
    X: pd.DataFrame,
    y: np.ndarray,
    problem_type: str,
    model_name: str,
    iterations: int = SEARCH_ITERATIONS,
) -> dict[str, Any] | None:
    """Randomised search around a model's defaults. None when not applicable."""
    grid = GRIDS.get(model_name)
    if grid is None or y is None or len(X) < SEARCH_FOLDS * 3:
        return None

    factory = data_tools.model_factory(problem_type, model_name)
    if factory is None:
        return None

    scoring = "f1_macro" if problem_type == "classification" else "r2"
    if problem_type == "classification" and pd.Series(y).value_counts().min() < SEARCH_FOLDS:
        return None

    settings = get_settings()
    try:
        search = RandomizedSearchCV(
            factory(),
            grid,
            n_iter=iterations,
            cv=SEARCH_FOLDS,
            scoring=scoring,
            random_state=settings.random_state,
            n_jobs=-1,
            error_score="raise" if False else np.nan,
        )
        search.fit(X, y)
    except Exception:
        return None

    if not np.isfinite(search.best_score_):
        return None
    return {
        "model": model_name,
        "params": {k: _plain(v) for k, v in search.best_params_.items()},
        "cv_score": float(search.best_score_),
        "metric": "macro F1" if problem_type == "classification" else "R²",
        "iterations": iterations,
        "folds": SEARCH_FOLDS,
    }


def _plain(value: Any) -> Any:
    """numpy scalars are not JSON-serialisable and end up in the DB."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 6)
    return value
