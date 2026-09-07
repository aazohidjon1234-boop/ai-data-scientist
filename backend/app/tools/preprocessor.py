"""The fitted preprocessing, kept so a saved model can be used again.

Training fits a scaler, decides fill values and one-hot column names, then
throws all of it away — only the estimator was persisted. That makes a
downloaded `.pkl` almost unusable: feeding it raw rows silently produces
nonsense, because the numbers are on a different scale and the columns are in a
different order.

This records every decision as plain JSON (no pickled sklearn objects, so an
upgrade cannot break loading) and replays them on new rows in the same order:
derive → select → fill → scale → encode → align columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ValidationError


@dataclass
class Preprocessor:
    features: list[str] = field(default_factory=list)          # final column order
    source_columns: list[str] = field(default_factory=list)    # raw inputs used
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    fill_values: dict[str, Any] = field(default_factory=dict)
    scaler_mean: dict[str, float] = field(default_factory=dict)
    scaler_scale: dict[str, float] = field(default_factory=dict)
    class_labels: list[str] | None = None
    engineered: list[dict[str, Any]] = field(default_factory=list)
    target: str | None = None
    problem_type: str = "classification"

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "source_columns": self.source_columns,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "fill_values": self.fill_values,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "class_labels": self.class_labels,
            "engineered": self.engineered,
            "target": self.target,
            "problem_type": self.problem_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Preprocessor":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    # ------------------------------------------------------------------
    def missing_columns(self, df: pd.DataFrame) -> list[str]:
        """Raw inputs the caller failed to supply. Derived columns are rebuilt."""
        derived = {spec.get("name") for spec in self.engineered}
        return [c for c in self.source_columns if c not in df.columns and c not in derived]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turn raw rows into the exact matrix the model was fitted on."""
        from .data_tools import _safe_feature_names, coerce_numeric_columns

        if self.engineered:
            from ..agents.feature_engineer import build_features

        frame, _ = coerce_numeric_columns(df.copy())

        if self.engineered:
            frame, _ = build_features(frame, self.engineered)

        missing = self.missing_columns(frame)
        if missing:
            raise ValidationError(
                "These columns are required but missing: " + ", ".join(missing[:10])
            )

        keep = [c for c in self.source_columns if c in frame.columns]
        frame = frame[keep].copy()

        # Fill with the values decided at training time — recomputing a median
        # from the new rows would shift the model's inputs.
        for column in frame.columns:
            if frame[column].isna().any() and column in self.fill_values:
                frame[column] = frame[column].fillna(self.fill_values[column])

        for column in self.numeric:
            if column not in frame.columns:
                continue
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].fillna(self.fill_values.get(column, 0.0))
            mean = self.scaler_mean.get(column)
            scale = self.scaler_scale.get(column)
            if mean is not None and scale:
                frame[column] = (frame[column] - mean) / scale

        categorical = [c for c in self.categorical if c in frame.columns]
        encoded = pd.get_dummies(frame, columns=categorical, dummy_na=False, dtype=int)
        encoded.columns = _safe_feature_names(encoded.columns)

        # A category unseen here leaves its column absent; a new category creates
        # one the model never saw. Both are resolved by aligning to the training
        # layout: missing become 0, unknown are dropped.
        aligned = encoded.reindex(columns=self.features, fill_value=0)
        return aligned.astype(float).fillna(0.0)

    def decode(self, predictions: np.ndarray) -> list[Any]:
        """Map encoded class indices back to the original labels."""
        values = np.asarray(predictions).ravel().tolist()
        if self.problem_type != "classification" or not self.class_labels:
            return values
        out = []
        for value in values:
            try:
                out.append(self.class_labels[int(value)])
            except (ValueError, IndexError, TypeError):
                out.append(value)
        return out
