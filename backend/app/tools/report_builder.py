"""tool: generate_report — assembles the full Markdown report.

Every number in the report comes from the analysis/training results that
the Python tools computed. Nothing is invented.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

TASK_LABEL = {"regression": "Regression", "classification": "Classification", "clustering": "Clustering"}


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def generate_report(
    dataset_name: str,
    analysis: dict[str, Any],
    training: dict[str, Any] | None,
    cleaned: dict[str, Any] | None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    profile = analysis.get("profile", {})
    missing = analysis.get("missing", {})
    stats = analysis.get("statistics", {})
    corr = analysis.get("correlation")
    outliers = analysis.get("outliers", {})
    problem = analysis.get("target_detection", {})
    problem_type = analysis.get("problem_type", "clustering")
    target = analysis.get("target_column")

    L: list[str] = []
    L.append(f"# Dataset Analysis Report — {dataset_name}")
    L.append("")
    L.append(f"_Generated {now} by the AI Data Scientist Agent. "
             "All numbers below are computed with pandas / scikit-learn._")
    L.append("")

    # 1. overview
    L.append("## 1. Dataset Overview")
    L.append("")
    L.append("| Property | Value |")
    L.append("|---|---|")
    L.append(f"| Rows | {profile.get('rows')} |")
    L.append(f"| Columns | {profile.get('columns')} |")
    L.append(f"| Numeric columns | {len(profile.get('numeric_columns', []))} |")
    L.append(f"| Categorical columns | {len(profile.get('categorical_columns', []))} |")
    L.append(f"| Duplicate rows | {profile.get('duplicate_rows')} |")
    L.append(f"| Missing values | {missing.get('total_missing')} ({missing.get('pct_missing')}%) |")
    L.append(f"| Problem type | {TASK_LABEL.get(problem_type, problem_type)} |")
    L.append(f"| Target column | {target or 'none (unsupervised)'} |")
    L.append("")
    if problem.get("reasoning"):
        L.append(f"> **How the task was decided:** {problem['reasoning']}")
        L.append("")

    # 2. data quality
    L.append("## 2. Data Quality")
    L.append("")
    if missing.get("per_column"):
        L.append("| Column | Missing | % | Imputation used |")
        L.append("|---|---|---|---|")
        for row in missing["per_column"][:20]:
            L.append(f"| {row['column']} | {row['missing']} | {row['pct']}% | {row['imputation']} |")
    else:
        L.append("No missing values were found.")
    L.append("")
    if cleaned and cleaned.get("operations"):
        L.append("**Cleaning operations applied:**")
        L.append("")
        for op in cleaned["operations"]:
            L.append(f"- {op}")
        L.append("")
    if outliers.get("columns"):
        L.append(f"**Outliers (IQR rule):** {outliers.get('total_outliers')} value(s) across "
                 f"{len(outliers['columns'])} column(s).")
        L.append("")

    # 3. preview
    preview = analysis.get("preview", {})
    if preview.get("rows"):
        L.append("## 3. Data Preview (first rows)")
        L.append("")
        cols = preview.get("columns", [])
        L.append("| " + " | ".join(str(c) for c in cols) + " |")
        L.append("|" + "---|" * len(cols))
        for row in preview["rows"][:10]:
            L.append("| " + " | ".join(_fmt(v) for v in row) + " |")
        L.append("")

    # 4. statistics
    L.append("## 4. Statistical Summary")
    L.append("")
    num_stats = {k: v for k, v in stats.items() if v.get("kind") == "numeric"}
    if num_stats:
        L.append("| Column | Mean | Median | Std | Min | Q1 | Q3 | Max |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k, v in num_stats.items():
            L.append(
                f"| {k} | {_fmt(v['mean'])} | {_fmt(v['median'])} | {_fmt(v['std'])} "
                f"| {_fmt(v['min'])} | {_fmt(v['q1'])} | {_fmt(v['q3'])} | {_fmt(v['max'])} |"
            )
        L.append("")
    cat_stats = {k: v for k, v in stats.items() if v.get("kind") == "categorical"}
    if cat_stats:
        L.append("### Categorical columns")
        L.append("")
        for k, v in cat_stats.items():
            top = ", ".join(f"{t['value']} ({t['count']})" for t in v["top_values"][:4])
            L.append(f"- **{k}** — {v['unique']} unique values; top: {top}")
        L.append("")

    # 5. correlations
    if corr and corr.get("top_pairs"):
        L.append("## 5. Strongest Correlations")
        L.append("")
        L.append("| Column A | Column B | r |")
        L.append("|---|---|---|")
        for p in corr["top_pairs"][:10]:
            L.append(f"| {p['a']} | {p['b']} | {p['corr']:.3f} |")
        L.append("")

    # 6. models
    if training:
        L.append(f"## 6. Model Comparison ({TASK_LABEL.get(problem_type, problem_type)})")
        L.append("")
        run_info = training.get("run_info", {})
        L.append(f"- Training rows: {run_info.get('n_train')} — Test rows: {run_info.get('n_test')}")
        L.append(f"- Features used: {run_info.get('n_features')}")
        if run_info.get("sampled"):
            L.append(f"- ⚠ Dataset was subsampled to {run_info.get('n_train_rows_used')} rows for training (speed limit).")
        L.append("")
        models = training.get("models", [])
        metric_cols = [
            ("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²"),
            ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
            ("silhouette", "Silhouette"), ("inertia", "Inertia"),
        ]
        present = [(k, label) for k, label in metric_cols
                   if any(m.get("metrics", {}).get(k) is not None for m in models)]
        header = ["Model", "Status", "Time (s)"] + [label for _, label in present]
        L.append("| " + " | ".join(header) + " |")
        L.append("|" + "---|" * len(header))
        for m in sorted(models, key=lambda x: x.get("rank", 99)):
            row = [
                f"**{m['name']}** ⭐" if m.get("is_best") else m["name"],
                m.get("status", "ok"),
                f"{m.get('training_seconds', 0):.2f}",
            ]
            for k, _ in present:
                v = m.get("metrics", {}).get(k)
                row.append(_fmt(v))
            L.append("| " + " | ".join(row) + " |")
        L.append("")

        best = next((m for m in models if m.get("is_best")), None)
        if best:
            L.append(f"### Best model: {best['name']}")
            L.append("")
            fm = best.get("metrics", {})
            parts = [f"{k}={_fmt(v)}" for k, v in fm.items() if k != "primary_metric" and v is not None and k != "confusion_matrix"]
            if parts:
                L.append(f"Test metrics: {', '.join(parts)}.")
            L.append("")
            fi = best.get("feature_importance") or []
            if fi:
                L.append("**Most important features:**")
                L.append("")
                L.append("| Feature | Importance |")
                L.append("|---|---|")
                for f in fi[:10]:
                    L.append(f"| {f['feature']} | {f['importance']} |")
                L.append("")

        L.append("## 7. AI Interpretation")
        L.append("")
        L.append(training.get("explanation") or "_No explanation generated._")
        L.append("")

    L.append("## 8. AI Interpretation of the Data")
    L.append("")
    L.append(analysis.get("explanation") or "_No explanation generated._")
    L.append("")
    L.append("---")
    L.append("_End of report. Produced by the AI Data Scientist Agent — every metric above was computed by the Python ML pipeline, not by the language model._")
    return "\n".join(L)
