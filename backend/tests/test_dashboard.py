"""Auto-composed dashboard: layout choices, filtering and KPI arithmetic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.dashboard import build_dashboard, dashboard_layout
from app.exceptions import ValidationError
from app.tools.query_tools import Filter


@pytest.fixture
def sales() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    n = 300
    return pd.DataFrame({
        "order_id": np.arange(1, n + 1),                       # identifier
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "region": rng.choice(["North", "South", "East"], n),
        "channel": rng.choice(["web", "store"], n),
        "revenue": rng.normal(100, 25, n).round(2),
        "constant": ["x"] * n,                                  # no variation
    })


def test_layout_separates_measures_from_dimensions(sales):
    layout = dashboard_layout(sales)
    assert "revenue" in layout["measures"]
    assert "region" in layout["dimensions"] and "channel" in layout["dimensions"]
    assert layout["primary_date"] == "order_date"


def test_layout_excludes_identifiers_and_constants(sales):
    layout = dashboard_layout(sales)
    everything = layout["measures"] + layout["dimensions"]
    assert "order_id" not in everything   # a running counter is not a metric
    assert "constant" not in everything   # nothing to show


def test_dashboard_builds_kpis_and_charts(sales):
    result = build_dashboard(sales)
    assert result["rows_shown"] == len(sales)
    assert result["kpis"] and result["charts"]
    assert all("title" in c and "data" in c for c in result["charts"])


def test_row_kpi_matches_the_frame(sales):
    rows = next(k for k in build_dashboard(sales)["kpis"] if k["label"] == "Rows")
    assert rows["value"] == len(sales)


def test_measure_totals_match_pandas(sales):
    result = build_dashboard(sales, measure="revenue")
    total = next(k for k in result["kpis"] if k["label"] == "Total revenue")
    average = next(k for k in result["kpis"] if k["label"] == "Average revenue")
    assert total["value"] == pytest.approx(sales["revenue"].sum())
    assert average["value"] == pytest.approx(sales["revenue"].mean())


def test_filters_narrow_every_number(sales):
    filters = [Filter(column="region", op="in", value=["North"])]
    result = build_dashboard(sales, filters=filters, measure="revenue")
    expected = sales[sales.region == "North"]
    assert result["rows_shown"] == len(expected)
    total = next(k for k in result["kpis"] if k["label"] == "Total revenue")
    assert total["value"] == pytest.approx(expected["revenue"].sum())


def test_empty_filter_result_is_reported_not_crashed(sales):
    result = build_dashboard(sales, filters=[Filter(column="region", op="in", value=["Mars"])])
    assert result["empty"] is True
    assert result["rows_shown"] == 0
    assert result["charts"] == []


def test_unknown_measure_is_rejected(sales):
    with pytest.raises(ValidationError, match="Unknown measure"):
        build_dashboard(sales, measure="nope")


def test_unknown_dimension_is_rejected(sales):
    with pytest.raises(ValidationError, match="Unknown dimension"):
        build_dashboard(sales, dimension="nope")


def test_filter_options_cover_the_dimensions(sales):
    options = {o["column"]: o["values"] for o in build_dashboard(sales)["filter_options"]}
    assert set(options["region"]) == set(sales["region"].unique())


def test_dataset_without_dates_still_builds():
    df = pd.DataFrame({"grp": ["a", "b"] * 50, "value": range(100)})
    result = build_dashboard(df)
    assert result["date_column"] is None
    assert result["charts"]  # bars/histogram still render


def test_dataset_without_numeric_measures_still_builds():
    df = pd.DataFrame({"grp": ["a", "b", "c"] * 30, "other": ["x", "y", "z"] * 30})
    result = build_dashboard(df)
    assert result["measure"] is None
    assert result["kpis"][0]["label"] == "Rows"
