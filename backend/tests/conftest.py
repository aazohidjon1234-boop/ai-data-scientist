"""Pytest fixtures: isolated temp dirs + test client.

Environment variables are set BEFORE the app is imported so the settings
and database engine pick up the temporary locations.
"""
from __future__ import annotations

import io
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

_TMP = pathlib.Path(os.environ.get("PYTEST_TMP", "/tmp/ads_test"))
_TMP.mkdir(parents=True, exist_ok=True)

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["UPLOAD_DIR"] = str(_TMP / "datasets")
os.environ["REPORT_DIR"] = str(_TMP / "reports")
os.environ["SAMPLE_DIR"] = str(_TMP / "samples")
os.environ["CORS_ORIGINS"] = "http://localhost:3000"

# fresh DB for each pytest session
db_file = _TMP / "test.db"
if db_file.exists():
    db_file.unlink()


# ---------------------------------------------------------------- fixtures
REGRESSION_CSV = """price,size,rooms,neighborhood
{rows}
"""


def _regression_rows(n: int = 60, seed: int = 1) -> str:
    rng = np.random.default_rng(seed)
    size = rng.integers(40, 400, n)
    rooms = rng.integers(1, 6, n)
    hood = rng.choice(["North", "South", "East"], n)
    price = (50 * size + 800 * rooms + rng.normal(0, 900, n)).round(0)
    lines = []
    for i in range(n):
        s = size[i] if i != 5 else ""  # one missing value
        p = f"{price[i]}" if i != 7 else ""  # one missing target
        lines.append(f"{p},{s},{rooms[i]},{hood[i]}")
    # 3 duplicate rows
    lines.extend(lines[:3])
    return "\n".join(lines)


def _classification_rows(n: int = 80, seed: int = 2) -> str:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(2 * x1 + 1.5 * x2)))
    label = (rng.random(n) < p).astype(int)
    cat = rng.choice(["high", "low"], n)
    lines = []
    for i in range(n):
        lines.append(
            f"{'Yes' if label[i] else 'No'},{x1[i]:.3f},{x2[i]:.3f},{cat[i]}"
            + ("" if i != 9 else "")
        )
    # one missing value
    lines[4] = lines[4].split(",")[0] + ",,," + lines[4].split(",")[3]
    return "outcome,x1,x2,plan\n" + "\n".join(lines)


def _clustering_rows(n: int = 60, seed: int = 3) -> str:
    rng = np.random.default_rng(seed)
    centers = [(0, 0, 5), (12, 12, 25), (22, 6, 45)]  # 3rd col correlates with group
    parts = [rng.normal(c[:2], 1.2, size=(n // 3, 2)) for c in centers]
    pts = np.vstack(parts)
    c3 = np.concatenate(
        [rng.normal(c[2], 1.5, n // 3) for c in centers]
    )
    lines = [f"{a:.2f},{b:.2f},{c:.2f}" for a, b, c in zip(pts[:, 0], pts[:, 1], c3)]
    return "col_a,col_b,col_c\n" + "\n".join(lines)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.main import app

    get_settings.cache_clear()  # pick up env set above

    with TestClient(app) as c:
        yield c


def upload_csv(client, content: str, name: str = "data.csv"):
    return client.post(
        "/api/datasets/upload",
        files={"file": (name, io.BytesIO(content.encode()), "text/csv")},
    )


@pytest.fixture
def regression_dataset_id(client) -> str:
    r = upload_csv(client, "price,size,rooms,neighborhood\n" + _regression_rows(), "reg.csv")
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.fixture
def classification_dataset_id(client) -> str:
    r = upload_csv(client, _classification_rows(), "clf.csv")
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.fixture
def clustering_dataset_id(client) -> str:
    r = upload_csv(client, _clustering_rows(), "clus.csv")
    assert r.status_code == 201, r.text
    return r.json()["id"]
