"""Convert numpy/pandas/plotly objects into plain JSON-safe structures."""
from __future__ import annotations

import json
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if np.isnan(f) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    if obj is None or isinstance(obj, (str, int, bool, float)):
        return obj
    return str(obj)


def fig_to_jsonable(fig) -> dict:
    """Serialize a Plotly figure to a plain JSON-serialisable dict."""
    return json.loads(fig.to_json())
