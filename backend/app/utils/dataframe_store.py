"""Reading CSVs, and a small in-process cache for parsed DataFrames.

Separator detection lives here rather than at the upload boundary because the
file is re-read on every later request (analysis, training, dashboard, ask).
When only the upload path detected ';', a semicolon file reported the right
column count on arrival and then came back as a single column for everything
that followed — Excel writes ';' in most non-US locales, so that silently broke
a large share of real files.
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Tried in order, only when the default read yields a single column.
ALTERNATE_SEPARATORS = (";", "\t", "|")
SNIFF_BYTES = 65_536


def _with_detected_separator(read, head: bytes, df: pd.DataFrame) -> pd.DataFrame:
    """Retry with another separator when the first read produced one column."""
    if df.shape[1] > 1:
        return df
    for sep in ALTERNATE_SEPARATORS:
        if sep.encode() not in head:
            continue
        try:
            alternative = read(sep)
        except Exception:  # noqa: BLE001 — pandas/csv raise many types
            continue
        if alternative.shape[1] > 1:
            return alternative
    return df


def read_csv_bytes(content: bytes, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), **kwargs)
    return _with_detected_separator(
        lambda sep: pd.read_csv(io.BytesIO(content), sep=sep, **kwargs),
        content[:SNIFF_BYTES],
        df,
    )


def read_csv_path(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, **kwargs)
    if df.shape[1] > 1:
        return df
    with open(path, "rb") as handle:
        head = handle.read(SNIFF_BYTES)
    return _with_detected_separator(lambda sep: pd.read_csv(path, sep=sep, **kwargs), head, df)


@lru_cache(maxsize=12)
def load_csv_cached(path: str) -> pd.DataFrame:
    return read_csv_path(path)


def clear_cache() -> None:
    load_csv_cached.cache_clear()
