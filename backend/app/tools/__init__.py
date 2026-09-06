"""Tool registry: maps tool names to callables (the agent's action space)."""
from __future__ import annotations

from . import analytics_tools, data_tools, query_tools, viz_tools

TOOL_REGISTRY = {
    name: getattr(data_tools, name)
    for name in (
        "analyze_dataset",
        "detect_missing_values",
        "clean_dataset",
        "detect_outliers",
        "generate_statistics",
        "generate_correlation_matrix",
        "detect_problem_type",
        "prepare_features",
        "train_model",
        "evaluate_model",
        "compare_models",
    )
} | {
    # analyst tools: ad-hoc querying, timelines and group comparison
    "run_query": query_tools.run_query,
    "pivot_table": query_tools.pivot_table,
    "describe_schema": query_tools.describe_schema,
    "time_series": analytics_tools.time_series,
    "compare_segments": analytics_tools.compare_segments,
    "detect_datetime_columns": analytics_tools.detect_datetime_columns,
}

TOOL_DESCRIPTIONS = {
    "analyze_dataset": "Profile shape, dtypes, duplicates and per-column summaries",
    "detect_missing_values": "Count missing values per column",
    "clean_dataset": "Remove empty columns/duplicates and impute missing values",
    "detect_outliers": "IQR-based outlier detection for numeric columns",
    "generate_statistics": "Mean, median, std, quartiles, min/max per column",
    "generate_correlation_matrix": "Pearson correlation matrix + strongest pairs",
    "create_visualization": "Build a Plotly figure (histogram, box, heatmap, scatter, bar)",
    "detect_problem_type": "Infer target column and regression/classification/clustering",
    "prepare_features": "Encoding, scaling, dropping unusable columns, train/test split",
    "train_model": "Fit a scikit-learn model on the training split",
    "evaluate_model": "Compute metrics on the held-out test split",
    "compare_models": "Rank models by their primary metric and pick the best",
    "generate_report": "Assemble the full Markdown analysis report",
    "describe_schema": "List columns with their role, cardinality and example values",
    "run_query": "Filter, group and aggregate the data from a validated query spec",
    "pivot_table": "Cross-tabulate two dimensions against a measure",
    "detect_datetime_columns": "Find columns that hold usable dates",
    "time_series": "Resample to a timeline: trend, growth, seasonality, anomalies",
    "compare_segments": "Compare groups and test whether the difference is significant",
    "build_query_spec": "Translate a natural-language question into a query spec",
    "generate_insights": "Rank what is noteworthy in a dataset without being asked",
}

__all__ = [
    "TOOL_REGISTRY",
    "TOOL_DESCRIPTIONS",
    "analytics_tools",
    "data_tools",
    "query_tools",
    "viz_tools",
]
