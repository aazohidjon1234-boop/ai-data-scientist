"""Dataset registration: validation, safe storage, sample handling."""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.database import Base
from ..core.security import safe_path
from ..exceptions import SampleNotFoundError, ValidationError
from ..models.db import Dataset, ModelResult


def ensure_dirs() -> None:
    settings = get_settings()
    for d in (
        Path(settings.upload_subdir),
        Path(settings.report_dir),
        Path(settings.model_dir),
        Path(settings.upload_dir),
    ):
        d.mkdir(parents=True, exist_ok=True)


def validate_dataframe(df: pd.DataFrame) -> None:
    settings = get_settings()
    if df.shape[0] == 0:
        raise ValidationError("The dataset has a header but no data rows.")
    if df.shape[0] > settings.max_rows:
        raise ValidationError(
            f"Dataset has {df.shape[0]} rows; the limit is {settings.max_rows} rows."
        )
    if df.shape[1] > settings.max_columns:
        raise ValidationError(
            f"Dataset has {df.shape[1]} columns; the limit is {settings.max_columns} columns."
        )
    if all(df[c].isna().all() for c in df.columns):
        raise ValidationError("All columns in the dataset are empty.")
    # dtype sanity: pandas gives us numeric/bool/object — everything else is unsupported
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) and not df[c].isna().all():
            df[c] = df[c].astype(str)


def parse_csv(content: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes; auto-detects comma/semicolon/tab separators.

    Excel in many locales (UZ, RU, DE, FR, ...) saves CSVs with ';' instead
    of ','. Without detection such files would be read as a single column.
    """
    import io

    settings = get_settings()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:  # noqa: BLE001 — csv module raises many types
        raise ValidationError(
            f"The file could not be parsed as CSV: {str(e)[:200]}"
        ) from e

    if df.shape[1] == 1:
        head = content[:65_536]
        for sep in (";", "\t"):
            if sep.encode() not in head:
                continue
            try:
                alt = pd.read_csv(io.BytesIO(content), sep=sep)
                if alt.shape[1] > 1:
                    df = alt
                    break
            except Exception:  # noqa: BLE001
                continue
    return df


def register_dataset(
    db: Session,
    df: pd.DataFrame,
    name: str,
    original_filename: str,
    csv_path: Path,
    source: str = "upload",
) -> Dataset:
    validate_dataframe(df)
    ds = Dataset(
        id=uuid.uuid4().hex,
        name=name,
        original_filename=original_filename,
        file_path=str(csv_path),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=[str(c) for c in df.columns],
        source=source,
        status="uploaded",
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


def get_dataset_or_404(db: Session, dataset_id: str) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        from ..exceptions import DatasetNotFoundError

        raise DatasetNotFoundError(f"Dataset '{dataset_id}' was not found.")
    return ds


def list_samples() -> list[dict[str, Any]]:
    settings = get_settings()
    sample_dir = Path(settings.sample_subdir)
    out: list[dict[str, Any]] = []
    if not sample_dir.exists():
        return out
    for csv_file in sorted(sample_dir.glob("*.csv")):
        meta_path = csv_file.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:  # noqa: BLE001
                meta = {}
        try:
            df = pd.read_csv(csv_file)
            rows, cols = int(df.shape[0]), int(df.shape[1])
        except Exception:  # noqa: BLE001
            continue
        out.append(
            {
                "name": csv_file.stem,
                "label": meta.get("label", csv_file.stem.replace("_", " ").title()),
                "description": meta.get(
                    "description", "Sample dataset for exploring the agent."
                ),
                "rows": rows,
                "columns": cols,
                "task": meta.get("task", "auto-detected"),
            }
        )
    return out


def load_sample(db: Session, name: str) -> Dataset:
    settings = get_settings()
    sample_dir = Path(settings.sample_subdir)
    src = safe_path(sample_dir, f"{name}.csv")
    if not src.exists():
        raise SampleNotFoundError(
            f"Sample '{name}' not found. Available: "
            + ", ".join(s["name"] for s in list_samples())
        )
    df = pd.read_csv(src)
    ds_id = uuid.uuid4().hex
    dest_dir = safe_path(settings.upload_subdir, ds_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{name}.csv"
    shutil.copyfile(src, dest)
    return register_dataset(db, df, name=name.title().replace("_", " "),
                            original_filename=f"{name}.csv", csv_path=dest, source="sample")


def clear_ml_state(db: Session, ds: Dataset) -> None:
    db.execute(ModelResult.__table__.delete().where(ModelResult.dataset_id == ds.id))
    ds.ml_run = None
    db.flush()
