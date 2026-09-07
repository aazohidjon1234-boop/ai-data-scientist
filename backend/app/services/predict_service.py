"""Score new rows with a model this app already trained.

The point of training is to use the result. Until now a run ended at a
downloadable `.pkl` that almost nobody could apply correctly, because the
scaler, the fill values and the one-hot column order lived only inside the run
that produced them.

The preprocessing is now stored with the run, so the same transformation is
replayed here and predictions come back on the original label scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.security import model_slug, safe_path
from ..exceptions import ValidationError
from ..tools.preprocessor import Preprocessor
from ..utils.jsonutils import to_jsonable
from .dataset_service import get_dataset_or_404

MAX_ROWS = 10_000


def _load(db: Session, dataset_id: str, model_name: str | None):
    ds = get_dataset_or_404(db, dataset_id)
    run = ds.ml_run or {}
    if not run:
        raise ValidationError("Train a model for this dataset first.")
    if run.get("problem_type") == "clustering":
        raise ValidationError("Clustering runs assign groups, they do not predict a target.")

    spec = (run.get("prep_report") or {}).get("preprocessor")
    if not spec:
        raise ValidationError(
            "This run predates prediction support. Train it again to enable predictions."
        )

    chosen = model_name or run.get("best_model")
    if not chosen:
        raise ValidationError("This run has no successful model to predict with.")

    path = safe_path(Path(get_settings().model_dir), ds.id, f"{model_slug(chosen)}.pkl")
    if not path.exists():
        raise ValidationError(
            f"The saved file for '{chosen}' is missing. Train again to regenerate it."
        )
    import joblib

    return ds, run, chosen, joblib.load(path), Preprocessor.from_dict(spec)


def required_inputs(db: Session, dataset_id: str) -> dict[str, Any]:
    """What a caller must supply, with an example row from the training data."""
    ds, run, chosen, _model, pre = _load(db, dataset_id, None)
    from ..utils.dataframe_store import load_csv_cached

    example: dict[str, Any] = {}
    try:
        df = load_csv_cached(ds.file_path)
        derived = {s.get("name") for s in pre.engineered}
        row = df.dropna().head(1)
        source = row if not row.empty else df.head(1)
        for column in pre.source_columns:
            if column in source.columns and column not in derived:
                example[column] = to_jsonable(source[column].iloc[0])
    except Exception:  # noqa: BLE001 — an example is a convenience, not a contract
        example = {}

    return {
        "dataset_id": ds.id,
        "model": chosen,
        "target": pre.target or run.get("target"),
        "problem_type": pre.problem_type,
        "required_columns": [c for c in pre.source_columns
                             if c not in {s.get("name") for s in pre.engineered}],
        "class_labels": pre.class_labels,
        "example_row": example,
    }


def predict(db: Session, dataset_id: str, rows: list[dict[str, Any]],
            model_name: str | None = None) -> dict[str, Any]:
    if not rows:
        raise ValidationError("Send at least one row to predict.")
    if len(rows) > MAX_ROWS:
        raise ValidationError(f"Send at most {MAX_ROWS} rows at a time.")

    ds, run, chosen, model, pre = _load(db, dataset_id, model_name)
    frame = pd.DataFrame(rows)
    X = pre.transform(frame)

    try:
        raw = model.predict(X)
    except Exception as exc:  # noqa: BLE001 — surface the model's own complaint
        raise ValidationError(f"The model could not score these rows: {str(exc)[:200]}") from exc

    predictions = pre.decode(raw)

    # Probabilities when the estimator offers them: a bare label hides how close
    # the call was, which matters more than the label for anything decisive.
    confidence: list[dict[str, float]] | None = None
    if pre.problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            labels = pre.class_labels or [str(c) for c in getattr(model, "classes_", [])]
            confidence = [
                {str(labels[i]) if i < len(labels) else str(i): round(float(p), 4)
                 for i, p in enumerate(row)}
                for row in proba
            ]
        except Exception:  # noqa: BLE001
            confidence = None

    return to_jsonable({
        "dataset_id": ds.id,
        "model": chosen,
        "target": pre.target or run.get("target"),
        "problem_type": pre.problem_type,
        "rows": len(frame),
        "predictions": predictions,
        "confidence": confidence,
    })


def predict_csv(db: Session, dataset_id: str, content: bytes,
                model_name: str | None = None) -> dict[str, Any]:
    from ..utils.dataframe_store import read_csv_bytes

    try:
        frame = read_csv_bytes(content)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"The file could not be parsed as CSV: {str(exc)[:200]}") from exc
    if frame.empty:
        raise ValidationError("The uploaded file has no rows.")
    result = predict(db, dataset_id, frame.head(MAX_ROWS).to_dict(orient="records"), model_name)
    result["truncated"] = len(frame) > MAX_ROWS
    return result
