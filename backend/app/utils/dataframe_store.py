"""Small in-process cache for parsed DataFrames (avoids re-reading CSVs)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=12)
def load_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clear_cache() -> None:
    load_csv_cached.cache_clear()
