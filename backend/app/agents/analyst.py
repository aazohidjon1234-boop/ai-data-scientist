"""Natural-language questions → validated QuerySpec → pandas → answer.

Mirrors the rule the rest of this project follows: the language model chooses
*what* to compute, Python computes it, and the narrative may only cite numbers
that came back from pandas.

Two routes, in priority order:

* **LLM** — asked to emit JSON for the closed QuerySpec grammar. Output is
  parsed, validated by Pydantic, then checked against the real columns. An
  invalid spec is fed back once with the error; a second failure falls through.
* **Heuristic** — matches column names and aggregation words directly from the
  question. Deterministic, needs no API key, and keeps the feature working when
  the LLM is unconfigured, rate-limited or down.

The answer sentence itself is always composed locally from the result table, so
it cannot contain a number pandas did not produce.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from ..exceptions import ValidationError
from ..tools.query_tools import (
    Aggregation,
    Filter,
    QuerySpec,
    describe_schema,
    run_query,
)
from ..tools.viz_tools import ACCENT, _wrap
from .llm_client import LLMClient

# Question wording -> aggregation. Ordered: the most specific phrasing wins.
_AGG_WORDS: list[tuple[tuple[str, ...], str]] = [
    (("how many", "number of", "count", "frequency", "occurrences"), "count"),
    (("distinct", "unique", "different"), "nunique"),
    (("total", "sum", "combined", "altogether"), "sum"),
    (("average", "avg", "mean", "typical"), "mean"),
    (("median",), "median"),
    (("spread", "deviation", "variance", "volatility"), "std"),
    (("maximum", "max", "highest", "largest", "biggest", "most expensive", "top"), "max"),
    (("minimum", "min", "lowest", "smallest", "cheapest"), "min"),
]

_ASCENDING_WORDS = ("lowest", "smallest", "cheapest", "worst", "least", "bottom", "minimum")

SPEC_INSTRUCTIONS = """You translate a question about a tabular dataset into a JSON query spec.

Reply with JSON ONLY — no prose, no markdown fence. Shape:
{
  "filters": [{"column": "<name>", "op": "=|!=|>|>=|<|<=|in|not_in|contains|between|is_null|not_null", "value": <any>}],
  "group_by": ["<name>"],
  "aggregations": [{"column": "<name or null>", "fn": "count|sum|mean|median|min|max|std|nunique", "alias": "<short name>"}],
  "select": ["<name>"],
  "sort_by": "<alias or column>",
  "ascending": false,
  "limit": 25
}

Rules:
- Use ONLY column names from the schema, copied exactly.
- "fn": "count" with "column": null counts rows.
- group_by is for the dimension asked about ("by region", "per product").
- Leave aggregations empty and use select+filters when raw rows are wanted.
- sort_by must be an aggregation alias or a grouped column.
- Never invent columns, values or filters that the question does not imply."""


# ---------------------------------------------------------------------------
# heuristic route
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _mentioned_columns(question: str, df: pd.DataFrame) -> list[str]:
    """Columns whose name appears in the question, longest match first."""
    asked = _normalise(question)
    hits = []
    for column in df.columns:
        token = _normalise(column)
        if token and re.search(rf"\b{re.escape(token)}\b", asked):
            hits.append((len(token), column))
    return [c for _, c in sorted(hits, reverse=True)]


def _detect_aggregation(question: str) -> str | None:
    asked = f" {_normalise(question)} "
    for words, fn in _AGG_WORDS:
        if any(f" {_normalise(w)} " in asked for w in words):
            return fn
    return None


def heuristic_spec(question: str, df: pd.DataFrame) -> QuerySpec:
    """Best-effort spec without any LLM. Always returns something runnable."""
    mentioned = _mentioned_columns(question, df)
    numeric = [c for c in mentioned if pd.api.types.is_numeric_dtype(df[c])]
    dimensions = [
        c for c in mentioned
        if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True) <= 15
    ]

    fn = _detect_aggregation(question) or ("mean" if numeric else "count")
    measure = next((c for c in numeric if c not in dimensions[:1]), None)

    if fn in {"sum", "mean", "median", "std", "min", "max"} and measure is None:
        # A numeric aggregation was asked for but no numeric column was named:
        # fall back to counting, which is always valid.
        fn = "count"

    group_by = dimensions[:1]
    if not group_by:
        candidates = [
            c for c in df.columns
            if (not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
            and 2 <= df[c].nunique(dropna=True) <= 15
        ]
        group_by = candidates[:1]

    aggregation = (
        Aggregation(fn="count", alias="row_count")
        if fn == "count"
        else Aggregation(column=measure, fn=fn, alias=f"{fn}_{measure}")
    )
    ascending = any(w in _normalise(question) for w in _ASCENDING_WORDS)
    return QuerySpec(
        group_by=group_by,
        aggregations=[aggregation],
        sort_by=aggregation.label(),
        ascending=ascending,
        limit=25,
    )


# ---------------------------------------------------------------------------
# LLM route
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any] | None:
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(body[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def llm_spec(question: str, df: pd.DataFrame, llm: LLMClient) -> QuerySpec | None:
    if not llm.enabled:
        return None
    schema = json.dumps(describe_schema(df), default=str)[:12_000]
    prompt = f"{SPEC_INSTRUCTIONS}\n\nSchema:\n{schema}\n\nQuestion: {question}\n\nJSON:"
    for attempt in range(2):
        raw = llm.complete(prompt, max_tokens=700, temperature=0.0)
        if not raw:
            return None
        payload = _extract_json(raw)
        if payload is None:
            prompt += "\n\nYour previous reply was not valid JSON. Reply with JSON only."
            continue
        try:
            spec = QuerySpec.model_validate(payload)
            run_query(df, spec)  # validates columns/types before we trust it
            return spec
        except (ValidationError, ValueError) as exc:
            if attempt == 0:
                prompt += f"\n\nThat spec was rejected: {exc}. Fix it and reply with JSON only."
    return None


# ---------------------------------------------------------------------------
# narrative + chart (composed from the result, never from the model)
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:,.2f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def summarise(result: dict[str, Any], question: str) -> str:
    rows = result.get("rows") or []
    if not rows:
        return "No rows match that question."

    spec = result.get("spec") or {}
    group_by = spec.get("group_by") or []
    metrics = [c for c in result["columns"] if c not in group_by]

    if not group_by:
        parts = [f"**{c}**: {_fmt(rows[0].get(c))}" for c in result["columns"]]
        return "Across " + f"{result['rows_after_filter']:,} rows — " + ", ".join(parts) + "."

    metric = metrics[0] if metrics else result["columns"][-1]
    dimension = group_by[0]
    top = rows[0]
    # The table is ordered by what was asked for, so "first row" means lowest
    # when the question asked for the lowest. Saying "leads" either way lies.
    ascending = bool(spec.get("ascending"))
    verb = f"has the lowest {metric}" if ascending else f"leads on {metric}"
    lines = [
        f"**{top.get(dimension)}** {verb} at {_fmt(top.get(metric))}"
        f" (out of {result['row_count']:,} groups, {result['rows_after_filter']:,} rows)."
    ]
    if len(rows) > 1:
        runners = ", ".join(f"{r.get(dimension)} ({_fmt(r.get(metric))})" for r in rows[1:4])
        lines.append(f"Then: {runners}." if ascending else f"Next: {runners}.")
    if result.get("truncated"):
        lines.append(f"Showing the first {len(rows)} of {result['row_count']:,} groups.")
    return " ".join(lines)


def result_chart(result: dict[str, Any]) -> dict[str, Any] | None:
    """Bar chart of the first metric per group — None when it would be noise."""
    rows = result.get("rows") or []
    spec = result.get("spec") or {}
    group_by = spec.get("group_by") or []
    if not rows or not group_by or len(rows) < 2:
        return None
    metrics = [c for c in result["columns"] if c not in group_by]
    if not metrics:
        return None
    metric, dimension = metrics[0], group_by[0]
    subset = rows[:20]
    values = [r.get(metric) for r in subset]
    if any(not isinstance(v, (int, float)) for v in values):
        return None
    fig = go.Figure(
        go.Bar(
            x=[str(r.get(dimension)) for r in subset],
            y=values,
            marker_color=ACCENT,
        )
    )
    fig.update_layout(xaxis_title=str(dimension), yaxis_title=str(metric))
    return _wrap(fig, "bar", f"{metric} by {dimension}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def answer_question(df: pd.DataFrame, question: str, llm: LLMClient | None = None) -> dict[str, Any]:
    """Answer an ad-hoc question about the dataset."""
    question = (question or "").strip()
    if not question:
        raise ValidationError("Ask a question about the data.")

    llm = llm or LLMClient()
    spec, route = llm_spec(question, df, llm), "llm"
    if spec is None:
        spec, route = heuristic_spec(question, df), "heuristic"

    result = run_query(df, spec)
    return {
        "question": question,
        "route": route,
        "answer": summarise(result, question),
        "table": result,
        "chart": result_chart(result),
        "tools_used": ["build_query_spec", "run_query"],
    }
