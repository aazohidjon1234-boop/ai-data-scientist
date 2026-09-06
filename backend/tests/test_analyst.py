"""Tests for the analyst layer: query spec, time series, segments, insights.

The query engine is the security-sensitive part — it is the one place a model's
output influences what runs against user data — so unknown columns, wrong types
and oversized groupings are all asserted to fail loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents.analyst import answer_question, heuristic_spec, summarise
from app.agents.insights import generate_insights
from app.exceptions import ValidationError
from app.tools.analytics_tools import (
    compare_segments,
    detect_datetime_columns,
    time_series,
)
from app.tools.query_tools import (
    Aggregation,
    Filter,
    QuerySpec,
    MAX_GROUPS,
    MAX_LIMIT,
    describe_schema,
    pivot_table,
    run_query,
)


@pytest.fixture
def sales() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "region": rng.choice(["North", "South", "East"], n),
            "channel": rng.choice(["web", "store"], n),
            "revenue": rng.normal(100, 20, n).round(2),
            "units": rng.integers(1, 20, n),
        }
    )


# --------------------------------------------------------------- query spec
def test_group_by_aggregation(sales):
    result = run_query(
        sales,
        QuerySpec(
            group_by=["region"],
            aggregations=[Aggregation(column="revenue", fn="mean", alias="avg")],
        ),
    )
    assert set(result["columns"]) == {"region", "avg"}
    assert result["row_count"] == sales["region"].nunique()
    # Default ordering puts the largest metric first.
    values = [r["avg"] for r in result["rows"]]
    assert values == sorted(values, reverse=True)


def test_aggregation_matches_pandas(sales):
    result = run_query(
        sales,
        QuerySpec(group_by=["region"], aggregations=[Aggregation(column="revenue", fn="sum", alias="total")]),
    )
    expected = sales.groupby("region")["revenue"].sum().to_dict()
    for row in result["rows"]:
        assert row["total"] == pytest.approx(expected[row["region"]])


def test_row_count_aggregation_without_column(sales):
    result = run_query(sales, QuerySpec(group_by=["channel"], aggregations=[Aggregation(fn="count")]))
    total = sum(r["row_count"] for r in result["rows"])
    assert total == len(sales)


def test_filters_narrow_the_frame(sales):
    result = run_query(
        sales,
        QuerySpec(
            filters=[Filter(column="region", op="in", value=["North"]), Filter(column="units", op=">=", value=10)],
            group_by=["region"],
            aggregations=[Aggregation(fn="count")],
        ),
    )
    expected = len(sales[(sales.region == "North") & (sales.units >= 10)])
    assert result["rows_after_filter"] == expected


def test_between_and_contains_filters(sales):
    between = run_query(sales, QuerySpec(filters=[Filter(column="units", op="between", value=[5, 10])], limit=MAX_LIMIT))
    assert between["rows_after_filter"] == len(sales[sales.units.between(5, 10)])
    contains = run_query(sales, QuerySpec(filters=[Filter(column="region", op="contains", value="nor")], limit=5))
    assert contains["rows_after_filter"] == len(sales[sales.region == "North"])


def test_unknown_column_is_rejected(sales):
    with pytest.raises(ValidationError, match="Unknown column"):
        run_query(sales, QuerySpec(group_by=["does_not_exist"]))


def test_unknown_filter_column_is_rejected(sales):
    with pytest.raises(ValidationError, match="Unknown column"):
        run_query(sales, QuerySpec(filters=[Filter(column="nope", op="=", value=1)]))


def test_mean_of_text_column_is_rejected(sales):
    with pytest.raises(ValidationError, match="not numeric"):
        run_query(sales, QuerySpec(aggregations=[Aggregation(column="region", fn="mean")]))


def test_too_many_groups_is_rejected():
    # Grouping by a near-unique key would return one row per record; the guard
    # exists so a mis-planned query cannot dump the whole dataset back.
    n = MAX_GROUPS * 2
    wide = pd.DataFrame({"order_id": [f"id-{i}" for i in range(n)], "revenue": np.arange(n, dtype=float)})
    with pytest.raises(ValidationError, match="more than"):
        run_query(wide, QuerySpec(group_by=["order_id"], aggregations=[Aggregation(fn="count")]))


def test_limit_is_capped():
    assert QuerySpec(limit=10_000).limit == MAX_LIMIT
    assert QuerySpec(limit=0).limit == 1


def test_empty_filter_result_is_not_an_error(sales):
    result = run_query(sales, QuerySpec(filters=[Filter(column="region", op="=", value="Atlantis")]))
    assert result["rows"] == [] and result["row_count"] == 0


def test_case_insensitive_column_rescue(sales):
    result = run_query(sales, QuerySpec(group_by=["REGION"], aggregations=[Aggregation(fn="count")]))
    assert result["row_count"] == sales["region"].nunique()


def test_describe_schema_classifies_columns(sales):
    schema = describe_schema(sales)
    kinds = {c["name"]: c["kind"] for c in schema["columns"]}
    assert kinds["revenue"] == "numeric"
    assert kinds["region"] == "categorical"
    assert schema["rows"] == len(sales)


def test_pivot_table_shape(sales):
    pivot = pivot_table(sales, index="region", columns="channel", values="revenue", aggfunc="mean")
    assert set(pivot["column_labels"]) == set(sales["channel"].unique())
    assert len(pivot["rows"]) == sales["region"].nunique()


# ------------------------------------------------------------- time series
def test_detect_datetime_columns(sales):
    assert detect_datetime_columns(sales) == ["order_date"]


def test_free_text_is_not_mistaken_for_dates():
    df = pd.DataFrame({"note": ["hello", "world", "some text", "more words", "again"]})
    assert detect_datetime_columns(df) == []


def test_rising_series_is_detected():
    n = 200
    df = pd.DataFrame(
        {"d": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
         "v": np.linspace(10, 100, n)}
    )
    result = time_series(df, "d", "v", agg="mean")
    assert result["trend"]["direction"] == "rising"
    assert result["trend"]["significant"] is True
    assert result["total_change_pct"] > 0


def test_flat_noise_is_not_reported_as_a_trend():
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame(
        {"d": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
         "v": rng.normal(50, 5, n)}
    )
    assert time_series(df, "d", "v", agg="mean")["trend"]["direction"] == "flat"


def test_spike_is_flagged_as_an_anomaly():
    n = 120
    values = np.full(n, 50.0)
    values[60] = 500.0
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=n, freq="D").astype(str), "v": values})
    result = time_series(df, "d", "v", agg="mean", freq="D")
    assert any(a["direction"] == "spike" for a in result["anomalies"])


def test_time_series_needs_a_real_date_column(sales):
    with pytest.raises(ValidationError):
        time_series(sales, "region", "revenue")


# ---------------------------------------------------------------- segments
def test_real_difference_is_significant():
    rng = np.random.default_rng(11)
    n = 300
    group = rng.choice(["a", "b"], n)
    value = np.where(group == "a", rng.normal(100, 5, n), rng.normal(60, 5, n))
    result = compare_segments(pd.DataFrame({"g": group, "v": value}), "g", "v")
    assert result["test"]["significant"] is True
    assert result["test"]["name"] == "Welch t-test"
    assert abs(result["test"]["effect_size"]["cohens_d"]) > 1


def test_identical_groups_are_not_significant():
    rng = np.random.default_rng(12)
    n = 300
    df = pd.DataFrame({"g": rng.choice(["a", "b", "c"], n), "v": rng.normal(50, 5, n)})
    result = compare_segments(df, "g", "v")
    assert result["test"]["significant"] is False
    assert "noise" in result["verdict"]


def test_three_groups_use_anova():
    rng = np.random.default_rng(13)
    n = 300
    group = rng.choice(["a", "b", "c"], n)
    value = np.select([group == "a", group == "b"], [rng.normal(90, 5, n), rng.normal(60, 5, n)], rng.normal(30, 5, n))
    result = compare_segments(pd.DataFrame({"g": group, "v": value}), "g", "v")
    assert result["test"]["name"] == "One-way ANOVA"
    assert 0 <= result["test"]["effect_size"]["eta_squared"] <= 1


def test_categorical_pair_uses_chi_square(sales):
    result = compare_segments(sales, "region", "channel")
    assert result["test"]["name"] == "Chi-square test of independence"
    assert result["kind"] == "categorical"


def test_single_group_is_rejected():
    with pytest.raises(ValidationError, match="fewer than two"):
        compare_segments(pd.DataFrame({"g": ["only"] * 10, "v": range(10)}), "g", "v")


# ---------------------------------------------------------------- insights
def test_insights_are_ranked_and_serialisable(sales):
    result = generate_insights(sales)
    scores = [i["importance"] for i in result["insights"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 100 for s in scores)
    assert isinstance(result["headline"], str)


def test_insights_flag_a_dominant_category():
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"flag": rng.choice(["yes", "no"], n, p=[0.95, 0.05]), "v": rng.normal(10, 2, n)})
    kinds = {i["kind"] for i in generate_insights(df)["insights"]}
    assert "imbalance" in kinds


def test_insights_flag_missing_data():
    df = pd.DataFrame({"a": [1.0 if i % 5 == 0 else None for i in range(200)], "b": range(200)})
    titles = " ".join(i["title"] for i in generate_insights(df)["insights"])
    assert "empty" in titles


def test_insights_survive_a_degenerate_frame():
    # One column, one value: no scanner should raise.
    result = generate_insights(pd.DataFrame({"only": [1] * 20}))
    assert isinstance(result["insights"], list)


# --------------------------------------------------------------- ask layer
def test_heuristic_spec_picks_dimension_and_measure(sales):
    spec = heuristic_spec("What is the average revenue by region?", sales)
    assert spec.group_by == ["region"]
    assert spec.aggregations[0].fn == "mean"
    assert spec.aggregations[0].column == "revenue"


def test_heuristic_spec_understands_counting(sales):
    spec = heuristic_spec("How many orders per channel?", sales)
    assert spec.aggregations[0].fn == "count"


def test_heuristic_spec_flips_sort_for_lowest(sales):
    assert heuristic_spec("Which region has the lowest revenue?", sales).ascending is True
    assert heuristic_spec("Which region has the highest revenue?", sales).ascending is False


def test_answer_question_without_llm(sales):
    result = answer_question(sales, "Which region has the highest average revenue?")
    assert result["route"] == "heuristic"
    assert result["table"]["rows"]
    # The narrative must name the top group from the computed table.
    assert str(result["table"]["rows"][0]["region"]) in result["answer"]


def test_answer_question_rejects_empty(sales):
    with pytest.raises(ValidationError):
        answer_question(sales, "   ")


def test_summary_wording_matches_sort_direction(sales):
    lowest = run_query(
        sales,
        QuerySpec(group_by=["region"], aggregations=[Aggregation(column="revenue", fn="mean", alias="m")],
                  sort_by="m", ascending=True),
    )
    assert "lowest" in summarise(lowest, "which region has the lowest revenue?")
