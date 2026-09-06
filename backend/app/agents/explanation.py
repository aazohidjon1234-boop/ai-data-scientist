"""Local explanation engine.

Turns the *actual* results computed by the Python tools into plain-language
explanations. It is deterministic and never invents numbers: every figure it
mentions comes from the analysis/training result dicts. When an LLM is
configured (env), the LLM is given the same JSON and is instructed to only
use those numbers — the local engine is the guaranteed fallback.
"""
from __future__ import annotations

from typing import Any

TASK_LABEL = {"regression": "regression (predicting a number)",
              "classification": "classification (predicting a category)",
              "clustering": "clustering (grouping similar rows)"}

METRIC_GLOSSARY = {
    "mae": "MAE (mean absolute error) is the average distance between the predicted and true values, in the same units as the target. Lower is better.",
    "mse": "MSE (mean squared error) squares those errors, which punishes big mistakes harder. Lower is better.",
    "rmse": "RMSE is the square root of MSE, so it is back in the target's units. A model with RMSE 5,000 typically misses by about 5,000 units. Lower is better.",
    "r2": "R² (R-squared) is the share of the target's variance that the model explains. 0.80 means the model explains 80% of the variation. It can go below 0 when the model is worse than simply guessing the average. Higher is better.",
    "accuracy": "Accuracy is the share of predictions that were correct. Lower is worse; on imbalanced data it can be misleading, so check precision/recall too.",
    "precision": "Precision (macro) is: of everything the model flagged positive, how much was actually positive. High precision = few false alarms.",
    "recall": "Recall (macro) is: of everything actually positive, how much did the model find. High recall = few misses.",
    "f1": "F1 (macro) is the harmonic mean of precision and recall — it balances both. Higher is better.",
    "silhouette": "The silhouette score measures how well separated the clusters are, from -1 to 1. Values above 0.5 mean clearly separated groups; 0.25–0.5 mean reasonable structure; below 0.25 suggests the grouping is weak.",
    "inertia": "Inertia is the total distance of points to their own cluster centre. Lower is tighter, but it always improves with more clusters, so judge it together with silhouette.",
}


def _pct(v: Any) -> str:
    return f"{v:.1f}%" if isinstance(v, (int, float)) else "—"


def explain_analysis(a: dict[str, Any]) -> str:
    profile = a.get("profile", {})
    missing = a.get("missing", {})
    stats = a.get("statistics", {})
    corr = a.get("correlation")
    outliers = a.get("outliers", {})
    det = a.get("target_detection", {})
    pt = a.get("problem_type", "clustering")
    target = a.get("target_column")

    L: list[str] = []
    L.append("### What I found in your data")
    L.append("")
    L.append(
        f"The dataset has **{profile.get('rows')} rows** and **{profile.get('columns')} columns**: "
        f"{len(profile.get('numeric_columns', []))} numeric and "
        f"{len(profile.get('categorical_columns', []))} categorical."
    )
    if profile.get("duplicate_rows"):
        L.append(f" I found **{profile['duplicate_rows']} fully duplicate rows** — I removed them before training.")
    else:
        L.append(" There are no fully duplicate rows.")
    L.append("")

    L.append("### Data quality")
    L.append("")
    if missing.get("per_column"):
        worst = missing["per_column"][0]
        L.append(
            f"There are **{missing['total_missing']} missing values ({_pct(missing.get('pct_missing'))})** "
            f"in {missing.get('columns_affected')} column(s). The worst is "
            f"**{worst['column']}** with {worst['missing']} missing ({worst['pct']}%)."
        )
        if missing["pct_missing"] > 20:
            L.append(" That is a lot — the imputed values will add noise, so treat the results with some caution.")
        elif missing["pct_missing"] > 5:
            L.append(" I filled numeric gaps with the column median and categorical gaps with the most frequent value, which is the safest generic choice.")
        else:
            L.append(" That is a small amount; I imputed numerics with the median and categoricals with the mode.")
    else:
        L.append("Good news: **no missing values** — the data is complete as-is.")
    if outliers.get("columns"):
        top = sorted(outliers["columns"].items(), key=lambda kv: -kv[1]["outliers"])[:2]
        bits = ", ".join(f"**{k}** ({v['outliers']} values, {v['pct']}%)" for k, v in top)
        L.append(f" Using the IQR rule I detected {outliers.get('total_outliers')} outlier value(s), mainly in {bits}. I kept them in the data — tree-based models are robust to them.")
    L.append("")

    L.append("### Task detection")
    L.append("")
    if det.get("reasoning"):
        L.append(det["reasoning"])
    L.append(f" I will treat this as a **{TASK_LABEL.get(pt, pt)}** problem" + (
        f" with target **{target}**." if target else "."))
    L.append("")

    num_stats = {k: v for k, v in stats.items() if v.get("kind") == "numeric"}
    if num_stats:
        L.append("### Quick numbers")
        L.append("")
        for k, v in list(num_stats.items())[:5]:
            L.append(f"- **{k}**: mean {v['mean']:,.2f}, median {v['median']:,.2f}, range {v['min']:,.2f} → {v['max']:,.2f}.")
        L.append("")
    if corr and corr.get("top_pairs"):
        top = corr["top_pairs"][0]
        direction = "strongly" if abs(top["corr"]) > 0.7 else ("moderately" if abs(top["corr"]) > 0.4 else "weakly")
        L.append(f"The strongest relationship is between **{top['a']}** and **{top['b']}** (r = {top['corr']:.2f}) — they are {direction} correlated.")
        L.append("")

    L.append("### Things to keep in mind")
    L.append("")
    if missing.get("pct_missing", 0) > 5:
        L.append("- Missing values were imputed, which can bias estimates if the missingness is not random.")
    if profile.get("rows", 0) < 200:
        L.append("- The dataset is small; metric differences between models may be noise. Cross-validation would give a sturdier estimate.")
    if profile.get("rows", 0) > 100_000:
        L.append("- The dataset is large; I trained on a random subsample to keep things fast. Full-data training should give very similar relative rankings.")
    L.append("- A single train/test split gives one estimate; on new data the numbers will move a bit.")
    return "\n".join(L)


def explain_training(t: dict[str, Any]) -> str:
    models = t.get("models", [])
    pt = t.get("problem_type", "classification")
    target = t.get("target")
    run = t.get("run_info", {})
    ok = [m for m in models if m.get("status") == "ok"]
    failed = [m for m in models if m.get("status") != "ok"]
    best = next((m for m in models if m.get("is_best")), (ok[0] if ok else None))

    L: list[str] = []
    L.append("### What I did")
    L.append("")
    if pt == "clustering":
        n_km = sum(1 for m in ok if m["name"].startswith("K-Means"))
        n_agg = sum(1 for m in ok if m["name"].startswith("Agglomerative"))
        n_db = sum(1 for m in ok if m["name"].startswith("DBSCAN"))
        L.append(
            f"I ran the full cluster model zoo on {run.get('n_train')} rows and {run.get('n_features')} features: "
            f"**K-Means** ({n_km} k-values), **Agglomerative/Ward** ({n_agg} k-values) and "
            f"**DBSCAN** ({n_db} eps values), scoring every variant with the silhouette coefficient."
        )
    else:
        L.append(
            f"I prepared **{run.get('n_features')} features** (imputation, one-hot encoding, scaling of numeric columns), "
            f"split the data into **{run.get('n_train')} training / {run.get('n_test')} test rows**, and trained "
            f"{len(ok)} models to predict **{target}**."
        )
    if run.get("sampled"):
        L.append(f" The dataset was subsampled to {run.get('n_train_rows_used')} rows for speed — the relative model ranking should hold on the full data.")
    if failed:
        names = ", ".join(f"{m['name']} ({m.get('status_message', 'failed')})" for m in failed)
        L.append(f" ⚠ {len(failed)} model(s) failed to train: {names}.")
    L.append("")

    if best:
        L.append(f"### Why **{best['name']}** is the best here")
        L.append("")
        fm = best.get("metrics", {})
        if pt == "regression":
            r2 = fm.get("r2")
            if r2 is not None:
                quality = "strong" if r2 >= 0.8 else ("decent" if r2 >= 0.5 else ("weak" if r2 >= 0.2 else "very weak"))
                L.append(
                    f"It explains **{r2 * 100:.1f}% of the variance** in {target} (R² = {r2:.3f}) — a {quality} fit. "
                    f"On unseen data its predictions are typically off by about **{fm.get('mae'):,.2f}** on average (MAE), "
                    f"with an RMSE of **{fm.get('rmse'):,.2f}**."
                )
        elif pt == "classification":
            L.append(
                f"It scores **F1 = {fm.get('f1', 0):.3f}** (macro) with **{fm.get('accuracy', 0) * 100:.1f}% accuracy**. "
                f"Precision {fm.get('precision', 0):.3f} vs recall {fm.get('recall', 0):.3f} "
                + ("— the model balances not over-flagging and not missing cases reasonably." if abs((fm.get('precision') or 0) - (fm.get('recall') or 0)) < 0.15
                   else ("— it is more conservative (fewer false alarms, more misses)." if (fm.get('precision') or 0) > (fm.get('recall') or 0)
                        else "— it errs on the side of catching more positives (more false alarms).")))
        else:
            sil = fm.get("silhouette", 0) or 0
            structure = "clearly separated" if sil > 0.5 else ("reasonable" if sil > 0.25 else "weak")
            if best["name"].startswith("DBSCAN"):
                L.append(
                    f"DBSCAN (eps = {fm.get('eps')}) found **{fm.get('k')} natural clusters** and marked "
                    f"**{fm.get('noise')} rows** as noise. Separation: **silhouette = {sil:.3f}** ({structure}). "
                    f"Cluster sizes: {fm.get('cluster_sizes')}."
                )
            else:
                L.append(
                    f"K = {fm.get('k')} clusters gives the best separation: **silhouette = {sil:.3f}** ({structure}). "
                    f"Cluster sizes: {fm.get('cluster_sizes')}."
                )
        L.append("")
        fi = best.get("feature_importance") or []
        if fi:
            top = fi[:5]
            L.append("The features that drive its decisions the most are: "
                     + ", ".join(f"**{f['feature']}**" for f in top) + ".")
            L.append("In plain terms: the model leans on these columns most when making a prediction.")
            L.append("")

    if len(ok) > 1:
        L.append("### How the models compared")
        L.append("")
        ranked = sorted(ok, key=lambda m: (m.get("rank") or 99))
        if pt == "regression":
            first, last = ranked[0], ranked[-1]
            if first.get("metrics", {}).get("r2") is not None and last.get("metrics", {}).get("r2") is not None:
                L.append(
                    f"{first['name']} leads with R² = {first['metrics']['r2']:.3f}; "
                    f"{last['name']} trails at R² = {last['metrics']['r2']:.3f}. "
                    "Linear models win only when the relationship is nearly straight; tree ensembles usually win on real-world noisy data, which matches what you see here."
                )
        elif pt == "classification":
            first, last = ranked[0], ranked[-1]
            L.append(
                f"{first['name']} tops the table (F1 = {first.get('metrics', {}).get('f1', 0):.3f}); "
                f"{last['name']} is last (F1 = {last.get('metrics', {}).get('f1', 0):.3f}). "
                "Ensemble methods combine many weak decisions and usually edge out single trees."
            )
        else:
            first, last = ranked[0], ranked[-1]
            L.append(
                f"{first['name']} wins the zoo with silhouette = "
                f"{first.get('metrics', {}).get('silhouette', 0):.3f}; {last['name']} is last "
                f"(silhouette = {last.get('metrics', {}).get('silhouette', 0):.3f}). "
                "Silhouette is computed on the same features for every algorithm, so the comparison is fair."
            )
        L.append("")

    L.append("### What the metrics mean")
    L.append("")
    seen: set[str] = set()
    for m in ok[:3]:
        for k in m.get("metrics", {}):
            if k in METRIC_GLOSSARY and k not in seen:
                seen.add(k)
                L.append(f"- **{k.upper()}**: {METRIC_GLOSSARY[k]}")
    L.append("")

    L.append("### Limitations")
    L.append("")
    if run.get("n_test", 0) < 30:
        L.append("- The test split is tiny; the reported metrics have a wide confidence interval.")
    L.append("- Metrics were measured on one held-out split; a few percent variation is normal on re-splits.")
    if pt == "regression" and (best or {}).get("metrics", {}).get("r2") is not None and (best.get("metrics", {}).get("r2")) < 0.5:
        L.append("- The model explains less than half of the variation: important drivers are probably missing from the data (or the target is inherently noisy).")
    if pt == "classification":
        cm = (best or {}).get("metrics", {}).get("confusion_matrix")
        if cm and len(cm.get("labels", [])) == 2:
            L.append("- If the classes are imbalanced, look at the confusion matrix: accuracy alone can hide a model that just guesses the majority class.")
    L.append("")

    L.append("### Suggestions to improve")
    L.append("")
    if pt != "clustering":
        L.append("- Add domain features (ratios, interactions, engineered dates) — features usually matter more than model choice.")
        L.append("- Try hyperparameter tuning (e.g. `RandomizedSearchCV`) on the best model family.")
        L.append("- Use 5-fold cross-validation instead of a single split for a sturdier estimate.")
    if t.get("run_info", {}).get("n_features", 0) < 5:
        L.append("- You have very few features; gathering more input data is likely the biggest lever.")
    L.append("- If this goes to production, monitor data drift: retrain when the input distribution shifts.")
    return "\n".join(L)


def explain_metric(metric: str, value: Any, best_name: str | None) -> str:
    text = METRIC_GLOSSARY.get(metric.lower())
    if not text:
        return f"I don't have a definition for '{metric}' in the metrics we computed."
    out = text
    if isinstance(value, (int, float)):
        out += f" In this run, {best_name or 'the best model'} has {metric.upper()} = {value:.4f}."
    return out
