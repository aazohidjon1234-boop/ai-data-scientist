"""Conversational layer: answers user questions about the analysis.

The answerer grounds every statement in the stored analysis/training
results. With an LLM configured, the LLM gets the same JSON under a strict
"only use these numbers" system prompt; otherwise a local intent engine
composes the reply from the same data.
"""
from __future__ import annotations

import re
from typing import Any

from ..agents.explanation import METRIC_GLOSSARY, explain_analysis, explain_training
from ..agents.llm_client import LLMClient, build_interpretation_prompt
from ..models.db import Dataset
from .ml_service import training_from_db

_METRIC_WORDS = ["r2", "r²", "rmse", "mae", "mse", "accuracy", "precision",
                 "recall", "f1", "silhouette", "inertia", "confusion"]


def _intent(message: str) -> tuple[str, str | None]:
    m = message.lower()
    for w in _METRIC_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", m):
            return "metric", w
    if any(k in m for k in ("why", "reason")) and any(k in m for k in ("best", "chosen", "choose", "select", "pick", "model")):
        return "best_model", None
    if any(k in m for k in ("improve", "better", "suggest", "recommend", "increase", "boost")):
        return "improve", None
    if any(k in m for k in ("feature", "importan", "predictor", "driver", "depend")):
        return "features", None
    if "missing" in m or "clean" in m or "imput" in m:
        return "missing", None
    if "correlat" in m:
        return "correlation", None
    if "outlier" in m:
        return "outliers", None
    if "cluster" in m:
        return "clustering", None
    if any(k in m for k in ("summar", "overview", "what did you find", "explain", "insight", "tell me")):
        return "summary", None
    return "fallback", None


def _local_answer(intent: str, arg: str | None, ds: Dataset, analysis, training: dict | None) -> tuple[str, list[str]]:
    tools_used: list[str] = []
    if analysis is None:
        return ("This dataset has not been analyzed yet — run the analysis first.", [])

    if intent == "summary":
        tools_used = ["analyze_dataset", "generate_statistics"]
        return explain_analysis(_analysis_context(ds, analysis)), tools_used

    if intent == "best_model":
        if training is None:
            return "I haven't trained any models for this dataset yet — start training first.", ["compare_models"]
        tools_used = ["compare_models"]
        best = next((m for m in training["models"] if m.get("is_best")), None)
        if best is None:
            return "No model succeeded in this run — check the model table for error messages.", tools_used
        lines = [f"The best model is **{best['name']}**." ]
        fm = best.get("metrics", {})
        for k, v in fm.items():
            if isinstance(v, (int, float)) and k != "primary_metric":
                lines[-1] += f" Its test {k.upper()} is {v:.4f}."
                break
        lines.append("")
        lines.append(explain_training(training))
        return "\n".join(lines), tools_used

    if intent == "improve":
        if training is None:
            training = {"problem_type": analysis.problem_type, "target": analysis.target_column,
                        "run_info": {}, "models": [], "explanation": ""}
        text = explain_training(training) if training.get("models") else ""
        m = re.search(r"### Suggestions to improve\n(.*?)(?:\n###|\Z)", text, re.S)
        if m:
            return m.group(1).strip(), ["compare_models", "generate_report"]
        return "Train models first; then I can give concrete improvement suggestions based on the results.", ["compare_models"]

    if intent == "features":
        tools_used = ["train_model", "evaluate_model"]
        if training:
            best = next((mm for mm in training["models"] if mm.get("is_best")), None)
            fi = (best or {}).get("feature_importance") or []
            if fi:
                top = ", ".join(f"**{f['feature']}** ({f['importance']})" for f in fi[:8])
                return (f"Based on {best['name']}'s learned importances, the most influential features are: {top}. "
                        "These are the columns the model relies on most when making predictions."), tools_used
        stats = (analysis.statistics or {})
        corr = analysis.correlation
        if corr and corr.get("top_pairs"):
            p = corr["top_pairs"][0]
            return (f"I haven't fitted a model yet, but from the correlation matrix the numeric column most related to "
                    f"**{(analysis.target_column or 'the target')}** is {p['a']} (r={p['corr']:.2f})."), tools_used
        return "No model has been trained yet, so feature importances are not available. Run training first.", tools_used

    if intent == "missing":
        tools_used = ["detect_missing_values", "clean_dataset"]
        missing = analysis.missing or {}
        if missing.get("per_column"):
            rows = "\n".join(
                f"- **{c['column']}**: {c['missing']} missing ({c['pct']}%) → filled with {c['imputation']}"
                for c in missing["per_column"][:10]
            )
            return (f"The dataset has {missing['total_missing']} missing values ({missing['pct_missing']}%) in "
                    f"{missing['columns_affected']} column(s):\n{rows}\n\n"
                    "Cleaning imputed them (median for numeric, mode for categorical) before training."), tools_used
        return "There are no missing values in this dataset — nothing to impute.", tools_used

    if intent == "correlation":
        tools_used = ["generate_correlation_matrix"]
        corr = analysis.correlation
        if not corr:
            return "There are fewer than two numeric columns, so no correlation matrix exists.", tools_used
        lines = ["The strongest correlations (Pearson r):"]
        for p in corr["top_pairs"][:5]:
            lines.append(f"- {p['a']} × {p['b']}: **r = {p['corr']:.2f}**")
        return "\n".join(lines), tools_used

    if intent == "outliers":
        tools_used = ["detect_outliers"]
        out = analysis.outliers or {}
        if out.get("columns"):
            lines = [f"I found {out['total_outliers']} outlier values (IQR rule) in {len(out['columns'])} column(s):"]
            for c, v in sorted(out["columns"].items(), key=lambda kv: -kv[1]["outliers"])[:5]:
                lines.append(f"- {c}: {v['outliers']} outliers ({v['pct']}%), outside [{v['bounds'][0]:.1f}, {v['bounds'][1]:.1f}]")
            return "\n".join(lines), tools_used
        return "No outliers detected with the IQR rule.", tools_used

    if intent == "clustering":
        tools_used = ["train_model", "evaluate_model"]
        if training and training.get("problem_type") == "clustering":
            ok = [m for m in training["models"] if m.get("status") == "ok"]
            if ok:
                best = max(ok, key=lambda m: m["metrics"].get("silhouette", -1))
                fm = best["metrics"]
                return (f"K-Means was run for k = {min(m['metrics'].get('k', 0) for m in ok)}…{max(m['metrics'].get('k', 0) for m in ok)}. "
                        f"Best: **k = {fm['k']}** with silhouette **{fm['silhouette']:.3f}** and cluster sizes {fm['cluster_sizes']}."
                        + (" The clusters are clearly separated." if fm["silhouette"] > 0.5 else " The structure is moderate — interpret the groups with care.")), tools_used
        return "Clustering has not been run on this dataset. If there is no target column, start training and I will run a K-Means search automatically.", tools_used

    if intent == "metric":
        tools_used = ["evaluate_model"]
        gloss = METRIC_GLOSSARY.get(arg or "")
        if not gloss:
            return f"I only computed these metrics: {', '.join(METRIC_GLOSSARY)}.", tools_used
        value = None
        best_name = None
        if training:
            best = next((mm for mm in training["models"] if mm.get("is_best")), None)
            if best:
                best_name = best["name"]
                value = (best.get("metrics") or {}).get(arg)
        out = gloss
        if isinstance(value, (int, float)):
            out += f" For {best_name}, {arg.upper()} = {value:.4f}."
        else:
            out += " (No model has been trained yet, so I can't quote a value.)"
        return out, tools_used

    # fallback
    tools_used = ["analyze_dataset"]
    p = analysis.profile or {}
    return (
        f"I can help you with this dataset ({p.get('rows')} rows × {p.get('columns')} columns, "
        f"task: {analysis.problem_type}). Try asking:\n"
        "- \"Summarize what you found\"\n- \"Why is that the best model?\"\n"
        "- \"Which features matter most?\"\n- \"What does R² mean?\"\n"
        "- \"What missing values did you fix?\""
    ), tools_used


def _analysis_context(ds: Dataset, analysis) -> dict:
    return {
        "profile": analysis.profile or {},
        "missing": analysis.missing or {},
        "statistics": analysis.statistics or {},
        "correlation": analysis.correlation,
        "outliers": analysis.outliers or {},
        "target_detection": {
            "reasoning": (analysis.profile or {}).get("_target_reasoning", ""),
        },
        "problem_type": analysis.problem_type,
        "target_column": analysis.target_column,
        "explanation": analysis.explanation or "",
    }


def answer_question(ds: Dataset, analysis, training: dict | None, message: str) -> dict[str, Any]:
    intent, arg = _intent(message)

    if analysis is not None:
        # stash the detection reasoning for context (not persisted elsewhere)
        pass

    llm = LLMClient()
    if llm.enabled and analysis is not None:
        context: dict[str, Any] = _analysis_context(ds, analysis)
        if training:
            context["training"] = {
                "problem_type": training["problem_type"],
                "best_model": training["best_model"],
                "models": training["models"],
                "run_info": training.get("run_info", {}),
            }
        reply = llm.complete(build_interpretation_prompt("chat", context, question=message))
        if reply:
            return {"reply": reply, "tool_calls": ["context_lookup"], "source": "llm"}

    reply, tools = _local_answer(intent, arg, ds, analysis, training)
    return {"reply": reply, "tool_calls": tools, "source": "local-engine"}
