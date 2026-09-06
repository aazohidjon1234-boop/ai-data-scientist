"""Python tools — the only place where math/ML actually happens.

The AI agent calls these tools and interprets their (JSON) output.
Every tool returns plain JSON-serialisable structures, so no number in
any AI-generated explanation can be invented: they all come from here.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..core.config import get_settings
from ..exceptions import ProcessingError, ValidationError
from ..utils.jsonutils import to_jsonable

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_ID_NAME_HINTS = ("id", "index", "uuid", "key", "code", "zip", "zipcode", "email", "name")
_TARGET_NAME_HINTS = (
    "target", "label", "outcome", "result", "prediction", "predicted",
    "response", "dep", "y", "churn", "price", "salary", "sales", "revenue",
    "survived", "default", "fraud", "win", "approve", "purchase", "buy",
)
_BINARY_VALUES = [
    frozenset({"yes", "no"}), frozenset({"true", "false"}), frozenset({"0", "1"}),
    frozenset({"good", "bad"}), frozenset({"churned", "not_churned"}),
]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
    ]


def _name_score(column: str) -> int:
    c = column.lower().strip()
    score = 0
    for hint in _TARGET_NAME_HINTS:
        if hint in c.split() or (len(hint) >= 4 and hint in c):
            score += 50
            break
    if any(hint in c.split() or c == hint for hint in _ID_NAME_HINTS):
        score -= 100
    return score


# --------------------------------------------------------------------------
# tool: analyze_dataset
# --------------------------------------------------------------------------

def analyze_dataset(df: pd.DataFrame) -> dict[str, Any]:
    settings = get_settings()
    numeric = numeric_columns(df)
    categorical = categorical_columns(df)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]

    col_info = []
    for c in df.columns:
        s = df[c]
        kind = "numeric" if pd.api.types.is_numeric_dtype(s) else (
            "boolean" if pd.api.types.is_bool_dtype(s) else "categorical"
        )
        col_info.append(
            {
                "name": str(c),
                "dtype": str(s.dtype),
                "kind": kind,
                "unique": int(s.nunique(dropna=True)),
                "missing": int(s.isna().sum()),
            }
        )

    return to_jsonable(
        {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "numeric_columns": numeric,
            "categorical_columns": categorical,
            "all_nan_columns": all_nan_cols,
            "duplicate_rows": int(df.duplicated().sum()),
            "columns_info": col_info,
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "preview_rows": settings.preview_rows,
        }
    )


# --------------------------------------------------------------------------
# tool: detect_missing_values
# --------------------------------------------------------------------------

def detect_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    rows = int(df.shape[0]) or 1
    missing = df.isna()
    per_column = []
    for c in df.columns:
        n = int(missing[c].sum())
        if n > 0:
            per_column.append(
                {
                    "column": str(c),
                    "missing": n,
                    "pct": round(100.0 * n / rows, 2),
                    "imputation": (
                        "median" if pd.api.types.is_numeric_dtype(df[c]) else "mode"
                    ),
                }
            )
    total = int(missing.sum().sum())
    return to_jsonable(
        {
            "total_missing": total,
            "pct_missing": round(100.0 * total / (rows * max(1, df.shape[1])), 2),
            "columns_affected": len(per_column),
            "per_column": sorted(per_column, key=lambda r: -r["missing"]),
            "all_nan_columns": [c for c in df.columns if df[c].isna().all()],
        }
    )


# --------------------------------------------------------------------------
# tool: clean_dataset
# --------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop fully-empty columns and duplicate rows, impute missing values."""
    ops: list[str] = []
    out = df.copy()

    all_nan = [c for c in out.columns if out[c].isna().all()]
    if all_nan:
        out = out.drop(columns=all_nan)
        ops.append(f"Removed {len(all_nan)} completely empty column(s): {', '.join(map(str, all_nan))}")

    dupes = int(out.duplicated().sum())
    if dupes:
        out = out.drop_duplicates()
        ops.append(f"Dropped {dupes} duplicate row(s)")

    imputed: dict[str, Any] = {}
    for c in out.columns:
        n_missing = int(out[c].isna().sum())
        if n_missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            fill = out[c].median()
            if pd.isna(fill):
                fill = 0
            out[c] = out[c].fillna(fill)
            imputed[str(c)] = {"value": float(fill) if not pd.isna(fill) else 0.0, "strategy": "median", "count": n_missing}
        else:
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "unknown"
            out[c] = out[c].fillna(str(fill))
            imputed[str(c)] = {"value": str(fill), "strategy": "mode", "count": n_missing}
    for info in imputed.values():
        info["value"] = to_jsonable(info["value"])
    if imputed:
        ops.append(
            f"Imputed {sum(i['count'] for i in imputed.values())} missing value(s) "
            f"in {len(imputed)} column(s) (median for numeric, mode for categorical)"
        )

    report = to_jsonable(
        {
            "operations": ops or ["No cleaning needed — data was already clean"],
            "rows_before": int(df.shape[0]),
            "rows_after": int(out.shape[0]),
            "columns_removed": all_nan,
            "imputed": imputed,
        }
    )
    return out, report


# --------------------------------------------------------------------------
# tool: detect_outliers
# --------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for c in numeric_columns(df):
        s = df[c].dropna()
        if s.nunique(dropna=True) < 8:
            continue  # not meaningful for low-cardinality numerics
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        if n_out:
            out[str(c)] = {
                "method": "IQR (1.5x)",
                "outliers": n_out,
                "pct": round(100.0 * n_out / max(1, len(s)), 2),
                "bounds": [float(lo), float(hi)],
            }
    total = sum(v["outliers"] for v in out.values())
    return to_jsonable({"columns": out, "total_outliers": total})


# --------------------------------------------------------------------------
# tool: generate_statistics
# --------------------------------------------------------------------------

def generate_statistics(df: pd.DataFrame) -> dict[str, Any]:
    stats = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            q = s.quantile([0.25, 0.5, 0.75])
            stats[str(c)] = {
                "kind": "numeric",
                "count": int(s.count()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std()),
                "min": float(s.min()),
                "q1": float(q.iloc[0]),
                "q3": float(q.iloc[2]),
                "max": float(s.max()),
            }
        else:
            top = s.value_counts(dropna=True).head(5)
            stats[str(c)] = {
                "kind": "categorical",
                "unique": int(s.nunique(dropna=True)),
                "top_values": [
                    {"value": str(k), "count": int(v)} for k, v in top.items()
                ],
            }
    return to_jsonable(stats)


# --------------------------------------------------------------------------
# tool: generate_correlation_matrix
# --------------------------------------------------------------------------

def generate_correlation_matrix(df: pd.DataFrame) -> dict[str, Any] | None:
    num = df[numeric_columns(df)]
    if num.shape[1] < 2:
        return None
    corr = num.corr()
    pairs: list[dict] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append({"a": cols[i], "b": cols[j], "corr": float(v)})
    pairs.sort(key=lambda p: -abs(p["corr"]))
    return to_jsonable(
        {
            "columns": cols,
            "matrix": corr.round(3).values.tolist(),
            "top_pairs": pairs[:10],
        }
    )


# --------------------------------------------------------------------------
# tool: detect_problem_type
# --------------------------------------------------------------------------

def detect_problem_type(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    candidates: list[dict[str, Any]] = []

    for c in df.columns:
        s = df[c]
        n_unique = int(s.nunique(dropna=True))
        if n_unique == 0:
            continue
        score = _name_score(str(c))
        is_bool = pd.api.types.is_bool_dtype(s)
        is_num = pd.api.types.is_numeric_dtype(s) and not is_bool

        kind = "boolean" if is_bool else ("numeric" if is_num else "categorical")
        if kind == "boolean":
            score += 45
        elif kind == "categorical":
            if n_unique == 2:
                score += 40
                vals = set(s.dropna().astype(str).str.lower().unique())
                if any(vals == b for b in _BINARY_VALUES):
                    score += 15
            elif n_unique <= 10:
                score += 10  # possible multiclass, weak evidence
            elif n_unique <= settings.max_cardinality:
                score += 5
            else:
                score -= 60  # high cardinality is a poor target
        elif kind == "numeric":
            # bare numerics are poor auto-targets without a name hint:
            # the user can still pick one explicitly
            score += 10 if n_unique > 10 else 5

        candidates.append(
            {
                "column": str(c),
                "kind": kind,
                "unique": n_unique,
                "score": int(score),
                "suggested_task": (
                    "classification"
                    if (kind != "numeric")
                    else ("regression" if n_unique > 5 else "classification")
                ),
            }
        )

    if target is not None:
        if target not in df.columns:
            raise ValidationError(f"Target column '{target}' does not exist in the dataset.")
        candidates = [c for c in candidates if c["column"] == target] + [
            c for c in candidates if c["column"] != target
        ]
        best = candidates[0]
        chosen_target, chosen_task = target, best["suggested_task"]
        auto = False
        reasoning = (
            f"Target column '{target}' was provided by the user. "
            f"It has {best['unique']} unique values, so the task is {chosen_task}."
        )
    else:
        ranked = sorted(candidates, key=lambda c: -c["score"])
        best = ranked[0] if ranked else None
        if best is None or best["score"] < 20:
            chosen_target, chosen_task, auto = None, "clustering", False
            reasoning = (
                "No column clearly acts as a target (no name hints, boolean, "
                "or low-cardinality type). The dataset will be treated as an "
                "unsupervised clustering problem."
            )
        else:
            chosen_target, chosen_task, auto = best["column"], best["suggested_task"], True
            reasoning = (
                f"Column '{best['column']}' looks like the target: "
                f"{best['kind']} type with {best['unique']} unique values "
                f"(detection score {best['score']}). "
                f"Task inferred as {chosen_task}."
            )

    return to_jsonable(
        {
            "problem_type": chosen_task,
            "target": chosen_target,
            "auto_detected": auto and chosen_target is not None,
            "confidence": "high" if (target is not None or (best and best["score"] >= 40)) else ("medium" if chosen_target else "low"),
            "reasoning": reasoning,
            "candidates": sorted(candidates, key=lambda c: -c["score"])[:10],
        }
    )


# --------------------------------------------------------------------------
# tool: prepare_features
# --------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    target: str | None,
    problem_type: str,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, Any]]:
    """Turn a dataframe into a model-ready X/y pair.

    `features` restricts which columns become X. Omitted (None) means "use
    everything except the target", which is the historical behaviour. An empty
    list is a user error, not a request for zero features, and is rejected.
    """
    settings = get_settings()
    if problem_type != "clustering" and target is None:
        raise ValidationError("A target column is required for supervised tasks.")
    if target is not None and target not in df.columns:
        raise ValidationError(f"Target column '{target}' is not in this dataset.")

    data = df.copy()

    requested: list[str] | None = None
    if features is not None:
        unknown = [c for c in features if c not in df.columns]
        if unknown:
            raise ValidationError(
                "These columns are not in the dataset: " + ", ".join(map(str, unknown[:10]))
            )
        # Selecting the target as a feature would leak it straight into X.
        requested = [c for c in dict.fromkeys(features) if c != target]
        if not requested:
            raise ValidationError(
                "Choose at least one input column that is not the target column."
            )
        keep = requested + ([target] if problem_type != "clustering" else [])
        data = data[keep]

    if problem_type != "clustering":
        y_raw = data.pop(target)
    else:
        y_raw = None

    dropped: list[str] = []
    constant = [c for c in data.columns if data[c].nunique(dropna=True) <= 1]
    for c in constant:
        data = data.drop(columns=[c])
        dropped.append(f"{c} (constant)")

    high_card: list[tuple[str, int]] = []
    for c in data.columns:
        if (
            not pd.api.types.is_numeric_dtype(data[c])
            and not pd.api.types.is_bool_dtype(data[c])
            and data[c].nunique(dropna=True) > settings.max_cardinality
        ):
            high_card.append((c, int(data[c].nunique(dropna=True))))
    for c, n_u in high_card:
        data = data.drop(columns=[c])
        dropped.append(f"{c} (high cardinality: {n_u} unique values)")

    # impute whatever is left
    for c in data.columns:
        if data[c].isna().any():
            if pd.api.types.is_numeric_dtype(data[c]):
                data[c] = data[c].fillna(data[c].median() if data[c].median() is not None and not pd.isna(data[c].median()) else 0)
            else:
                mode = data[c].mode(dropna=True)
                data[c] = data[c].fillna(str(mode.iloc[0]) if len(mode) else "unknown")

    num_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    cat_cols = [c for c in data.columns if c not in num_cols]

    scaled_cols = []
    if num_cols:
        scaler = StandardScaler()
        data[num_cols] = scaler.fit_transform(data[num_cols])
        scaled_cols = num_cols

    dummied = pd.get_dummies(data, columns=cat_cols, dummy_na=False, dtype=int)
    encoded = [c for c in dummied.columns if c not in num_cols]

    if y_raw is not None:
        y = y_raw
        class_labels = None
        if problem_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
            class_labels = [str(v) for v in le.classes_]
        else:
            y = pd.to_numeric(y_raw, errors="coerce")
            if y.isna().any():
                raise ValidationError(
                    f"Target column '{target}' contains non-numeric values; "
                    "regression targets must be numeric."
                )
            y = y.to_numpy(dtype=float)
    else:
        y = None
        class_labels = None

    report = to_jsonable(
        {
            "n_features": int(dummied.shape[1]),
            "features": list(map(str, dummied.columns)),
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "encoded_columns": encoded,
            "scaled_columns": scaled_cols,
            "dropped_columns": dropped or ["none"],
            "class_labels": class_labels,
            # What the user asked for, before automatic dropping — so the UI can
            # show that a chosen column was discarded and why.
            "selected_by_user": requested,
            "source_columns": sorted(set(num_cols) | set(cat_cols)),
        }
    )
    return dummied, y, report


# --------------------------------------------------------------------------
# tool: train_model / evaluate_model / compare_models
# --------------------------------------------------------------------------

def _xgb_regressor() -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as e:  # pragma: no cover
        raise ProcessingError("XGBoost is not installed (pip install xgboost)") from e
    return XGBRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=6,
        tree_method="hist", n_jobs=-1, random_state=42,
    )


def _lgbm_regressor() -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:  # pragma: no cover
        raise ProcessingError("LightGBM is not installed (pip install lightgbm)") from e
    return LGBMRegressor(
        n_estimators=300, learning_rate=0.08, num_leaves=31,
        n_jobs=-1, random_state=42, importance_type="gain", verbose=-1,
    )


def _xgb_classifier() -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as e:  # pragma: no cover
        raise ProcessingError("XGBoost is not installed (pip install xgboost)") from e
    return XGBClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=6,
        tree_method="hist", n_jobs=-1, random_state=42,
    )


def _lgbm_classifier() -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:  # pragma: no cover
        raise ProcessingError("LightGBM is not installed (pip install lightgbm)") from e
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.08, num_leaves=31,
        n_jobs=-1, random_state=42, importance_type="gain", verbose=-1,
    )


REGRESSION_MODELS: dict[str, Callable[[], Any]] = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=10, min_samples_leaf=3),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1),
    "Extra Trees": lambda: ExtraTreesRegressor(n_estimators=100, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
    "Hist Gradient Boosting": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08),
    "XGBoost": _xgb_regressor,
    "LightGBM": _lgbm_regressor,
    "KNN": lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    "SVR": lambda: SVR(kernel="rbf", C=1.0),
}

CLASSIFICATION_MODELS: dict[str, Callable[[], Any]] = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=5),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=100, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),
    "XGBoost": _xgb_classifier,
    "LightGBM": _lgbm_classifier,
    "SVM": lambda: SVC(kernel="rbf"),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Naive Bayes": lambda: GaussianNB(),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=100, learning_rate=0.5),
}

# Kernel methods are O(n^2..3): cap the rows they fit on (features are already scaled).
_MAX_FIT_ROWS = {"SVR": 15_000, "SVM": 15_000}

_PRIMARY_METRIC = {
    "regression": ("r2", "higher"),
    "classification": ("f1", "higher"),
    "clustering": ("silhouette", "higher"),
}


def train_model(
    model_name: str,
    problem_type: str,
    X: pd.DataFrame,
    y: np.ndarray | None,
    random_state: int = 42,
) -> tuple[Any, float]:
    """Train a single model; returns (model, elapsed_seconds).

    Kernel methods (SVR/SVM) are capped at _MAX_FIT_ROWS rows — they do not
    scale beyond that; a random subsample keeps them honest and fast.
    """
    registry = REGRESSION_MODELS if problem_type == "regression" else CLASSIFICATION_MODELS
    if model_name not in registry:
        raise ValidationError(f"Unknown model '{model_name}' for {problem_type}.")
    model = registry[model_name]()
    fit_X = X
    cap = _MAX_FIT_ROWS.get(model_name)
    if cap is not None and len(X) > cap:
        fit_X = X.sample(n=cap, random_state=random_state)
    t0 = time.perf_counter()
    model.fit(fit_X, y)
    return model, time.perf_counter() - t0


def evaluate_model(
    model: Any,
    model_name: str,
    problem_type: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_labels: list[str] | None = None,
) -> dict[str, Any]:
    if problem_type == "regression":
        pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        mse = float(mean_squared_error(y_test, pred))
        return to_jsonable(
            {
                "mae": mae,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(y_test, pred)),
                "primary_metric": "r2",
            }
        )
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=np.unique(np.concatenate([y_test, pred])) if len(np.unique(np.concatenate([y_test, pred]))) > 0 else None)
    labels = class_labels or [str(i) for i in range(cm.shape[0])]
    if len(labels) < cm.shape[0]:
        labels = [str(i) for i in range(cm.shape[0])]
    return to_jsonable(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist(),
                "support": cm.sum(axis=1).tolist(),
            },
            "primary_metric": "f1",
        }
    )


def evaluate_clustering(
    X: pd.DataFrame, k: int, random_state: int = 42
) -> tuple[dict[str, Any], KMeans]:
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0
    sizes = np.bincount(labels).tolist()
    metrics = to_jsonable(
        {
            "k": int(k),
            "silhouette": sil,
            "inertia": float(km.inertia_),
            "cluster_sizes": [int(s) for s in sizes],
            "primary_metric": "silhouette",
        }
    )
    return metrics, km


def evaluate_agglomerative(
    X: pd.DataFrame, k: int, random_state: int = 42
) -> tuple[dict[str, Any], AgglomerativeClustering]:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X)
    sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0
    sizes = np.bincount(labels).tolist()
    metrics = to_jsonable(
        {
            "k": int(k),
            "silhouette": sil,
            "inertia": float(getattr(model, "inertia_", 0.0)),
            "cluster_sizes": [int(s) for s in sizes],
            "primary_metric": "silhouette",
        }
    )
    return metrics, model


DBSCAN_EPS_VALUES = (0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def evaluate_dbscan(
    X: pd.DataFrame, eps: float, min_samples: int = 5, random_state: int = 42
) -> tuple[dict[str, Any], DBSCAN]:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    real_labels = labels[labels != -1]
    n_noise = int((labels == -1).sum())
    n_clusters = int(len(np.unique(real_labels)))
    if n_clusters > 1:
        sil = float(silhouette_score(X, labels))  # noise counts as its own cluster
    else:
        sil = -1.0
    sizes = [int((labels == c).sum()) for c in sorted(np.unique(real_labels))]
    metrics = to_jsonable(
        {
            "k": n_clusters,
            "eps": float(eps),
            "silhouette": sil,
            "cluster_sizes": sizes,
            "noise": n_noise,
            "primary_metric": "silhouette",
        }
    )
    return metrics, model


def feature_importance(model: Any, X: pd.DataFrame, limit: int = 15) -> list[dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        imps = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_))
        if coef.ndim == 2:
            coef = coef.sum(axis=0)
        total = coef.sum()
        imps = coef / total if total > 0 else coef
    else:
        return []
    order = np.argsort(-imps)[:limit]
    return to_jsonable(
        [{"feature": str(X.columns[i]), "importance": round(float(imps[i]), 4)} for i in order]
    )


def compare_models(results: list[dict[str, Any]], problem_type: str) -> dict[str, Any]:
    key, direction = _PRIMARY_METRIC.get(problem_type, ("f1", "higher"))
    ok = [r for r in results if r.get("status") == "ok" and r.get("metrics")]
    if not ok:
        return {"ranked": [], "best": None, "metric_key": key}
    ok = sorted(
        ok,
        key=lambda r: r["metrics"].get(key) if r["metrics"].get(key) is not None else -1e18,
        reverse=(direction == "higher"),
    )
    for i, r in enumerate(ok):
        r["rank"] = i + 1
    return {
        "ranked": [
            {
                "rank": r["rank"],
                "name": r["name"],
                "primary_metric": r["metrics"].get(key),
                "metrics": r["metrics"],
                "training_seconds": r.get("training_seconds", 0.0),
            }
            for r in ok
        ],
        "best": ok[0]["name"],
        "metric_key": key,
    }


def split_data(
    X: pd.DataFrame,
    y: np.ndarray | None,
    problem_type: str,
    random_state: int = 42,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None, np.ndarray | None]:
    if problem_type == "clustering" or y is None:
        return X, X, None, None
    stratify = y if problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test
