"""DataScientistAgent — orchestrates the Python tools like a data scientist.

Flow (mirrors how a human DS would work):
    inspect -> quality check -> statistics -> correlations -> outliers ->
    target/task detection -> clean -> prepare features -> train models ->
    evaluate -> compare -> explain

The agent decides the *plan* from observations (e.g. skips cleaning when
the data is already clean) and records every tool call in a trace that the
UI renders as an activity timeline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import get_settings
from ..exceptions import ProcessingError, ValidationError
from ..tools import data_tools, viz_tools
from .base import AgentTrace, ToolTimer
from .explanation import explain_analysis, explain_training
from .llm_client import LLMClient, build_interpretation_prompt

K_RANGE = (2, 9)  # K-Means search range


class DataScientistAgent:
    # Human-readable stage names for the live progress indicator.
    STAGE_LABELS = {
        "analyze_dataset": "Reading the file",
        "detect_missing_values": "Checking missing values",
        "clean_dataset": "Cleaning the data",
        "detect_outliers": "Looking for outliers",
        "generate_statistics": "Computing statistics",
        "generate_correlation_matrix": "Measuring correlations",
        "detect_problem_type": "Deciding the task",
        "prepare_features": "Preparing features",
        "train_model": "Training models",
        "evaluate_model": "Evaluating on held-out data",
        "compare_models": "Comparing models",
        "create_visualization": "Drawing charts",
        "generate_report": "Writing the explanation",
    }

    def __init__(self, df: pd.DataFrame, model_dir: Path | None = None,
                 dataset_id: str | None = None):
        self.settings = get_settings()
        self.df = df
        self.model_dir = model_dir  # when set, every trained model is persisted as .pkl
        self.llm = LLMClient()
        # Only set for real runs; tests construct the agent without it.
        self.dataset_id = dataset_id

    def _step(self, trace, tool: str, args: dict[str, Any]) -> ToolTimer:
        """A timed tool call that also publishes live progress."""
        return ToolTimer(trace, tool, args, dataset_id=self.dataset_id,
                         label=self.STAGE_LABELS.get(tool, tool))

    # ------------------------------------------------------------------ #
    # ANALYSIS
    # ------------------------------------------------------------------ #
    def analyze(self, target_override: str | None = None) -> dict[str, Any]:
        df = self.df
        trace = AgentTrace()
        settings = self.settings

        with self._step(trace, "analyze_dataset", {"columns": len(df.columns)}) as t:
            profile = data_tools.analyze_dataset(df)
            t.ok(
                f"{profile['rows']} rows × {profile['columns']} columns; "
                f"{len(profile['numeric_columns'])} numeric, "
                f"{len(profile['categorical_columns'])} categorical, "
                f"{profile['duplicate_rows']} duplicates"
            )

        with self._step(trace, "detect_missing_values", {}) as t:
            missing = data_tools.detect_missing_values(df)
            worst = missing["per_column"][0] if missing["per_column"] else None
            obs = (
                f"{missing['total_missing']} missing values ({missing['pct_missing']}%)"
                + (f"; worst: {worst['column']} ({worst['pct']}%)" if worst else "")
            )
            t.ok(obs)

        with self._step(trace, "generate_statistics", {}) as t:
            statistics = data_tools.generate_statistics(df)
            t.ok(f"Computed summary statistics for {len(statistics)} columns")

        with self._step(trace, "generate_correlation_matrix", {}) as t:
            correlation = data_tools.generate_correlation_matrix(df)
            if correlation:
                top = correlation["top_pairs"][0]
                t.ok(f"Correlation over {len(correlation['columns'])} numeric columns; strongest pair {top['a']}×{top['b']} (r={top['corr']:.2f})")
            else:
                t.ok("Fewer than 2 numeric columns — no correlation matrix", status="skipped")

        with self._step(trace, "detect_outliers", {}) as t:
            outliers = data_tools.detect_outliers(df)
            t.ok(f"{outliers['total_outliers']} outlier values across {len(outliers['columns'])} columns (IQR rule)")

        with self._step(trace, "detect_problem_type", {"target": target_override}) as t:
            detection = data_tools.detect_problem_type(df, target=target_override)
            t.ok(detection["reasoning"])

        target = detection["target"]
        problem_type = detection["problem_type"]

        # --- visuals (computed, not decorative) ---
        figures: list[dict[str, Any]] = []
        with self._step(trace, "create_visualization", {"kinds": ["histogram", "box", "bar", "missing", "scatter", "heatmap"]}) as t:
            figures += viz_tools.create_histograms(df)
            box = viz_tools.create_box_plot(df)
            if box:
                figures.append(box)
            figures += viz_tools.create_bar_charts(df)
            miss_fig = viz_tools.create_missing_bar(df)
            if miss_fig:
                figures.append(miss_fig)
            figures += viz_tools.create_scatter(df, correlation, target)
            heat = viz_tools.create_correlation_heatmap(correlation)
            if heat:
                figures.append(heat)
            target_dist = viz_tools.create_target_distribution(df, target)
            if target_dist:
                figures.append(target_dist)
            t.ok(f"Built {len(figures)} Plotly figures from real data")

        preview = self._preview(df)

        analysis = {
            "target_column": target,
            "problem_type": problem_type,
            "target_detection": detection,
            "profile": profile,
            "missing": missing,
            "statistics": statistics,
            "correlation": correlation,
            "outliers": outliers,
            "preview": preview,
            "figures": figures,
            "trace": trace.to_dict(),
        }

        plan = self._analysis_plan(profile, missing, detection)
        analysis["plan"] = plan

        # --- interpretation (LLM if configured, else local engine) ---
        with self._step(trace, "generate_report", {"kind": "explanation"}) as t:
            local_text = explain_analysis(analysis)
            llm_text = None
            if self.llm.enabled:
                llm_text = self.llm.complete(build_interpretation_prompt("analysis", analysis))
            analysis["explanation"] = llm_text or local_text
            analysis["explanation_source"] = "llm" if llm_text else "local-engine"
            t.ok(f"Explanation generated ({len(analysis['explanation'])} chars, source: {analysis['explanation_source']})")

        analysis["trace"] = trace.to_dict()
        return analysis

    # ------------------------------------------------------------------ #
    # TRAINING
    # ------------------------------------------------------------------ #
    def train(
        self,
        target: str | None = None,
        problem_type: str | None = None,
        k_range: tuple[int, int] | None = None,
        features: list[str] | None = None,
        drop_outliers: bool = False,
        tune: bool = False,
        impute_numeric: str = "median",
        impute_categorical: str = "mode",
    ) -> dict[str, Any]:
        settings = self.settings
        trace = AgentTrace()
        t0 = time.perf_counter()

        with self._step(trace, "analyze_dataset", {"purpose": "reload for training"}) as t:
            profile = data_tools.analyze_dataset(self.df)
            t.ok(f"{profile['rows']} rows × {profile['columns']} columns")

        with self._step(trace, "clean_dataset", {}) as t:
            clean_df, clean_report = data_tools.clean_dataset(self.df)
            t.ok("; ".join(clean_report["operations"]))

        with self._step(trace, "detect_problem_type", {"target": target}) as t:
            detection = data_tools.detect_problem_type(clean_df, target=target)
            if problem_type:
                detection["problem_type"] = problem_type
                detection["reasoning"] += f" (Task type overridden to {problem_type} by the user.)"
            if target is None and not problem_type and detection["target"] is None:
                detection["problem_type"] = "clustering"
            t.ok(detection["reasoning"])

        ptype = detection["problem_type"]
        tcol = detection["target"]
        if ptype == "clustering":
            tcol = None  # unsupervised: nothing may leak in as a feature
        if ptype != "clustering" and tcol is None:
            raise ValidationError(
                "No target column could be detected and none was provided. "
                "Choose a target column (or select 'Clustering' as the task)."
            )

        # Optionally drop rows that sit outside 1.5xIQR on a numeric column.
        # Only ever applied when the caller asked for it, because throwing away
        # observations is a modelling decision, not a cleaning detail.
        outlier_report: dict[str, Any] | None = None
        if drop_outliers and ptype != "clustering":
            from .improver import outlier_mask

            with self._step(trace, "detect_outliers", {"action": "drop rows"}) as t:
                inputs = [c for c in clean_df.columns if c != tcol]
                mask = outlier_mask(clean_df, inputs)
                removed = int(mask.sum())
                if removed and removed < len(clean_df) * 0.5:
                    clean_df = clean_df[~mask]
                    outlier_report = {"rows_removed": removed, "rows_left": int(len(clean_df))}
                    t.ok(f"Removed {removed} outlier rows; {len(clean_df)} left")
                else:
                    outlier_report = {"rows_removed": 0, "rows_left": int(len(clean_df))}
                    t.ok("No rows removed (none flagged, or too many to drop safely)")

        # downsample huge datasets for training speed (ranking is stable)
        work = clean_df
        sampled = False
        n_used = len(work)
        if n_used > settings.max_train_rows:
            work = work.sample(n=settings.max_train_rows, random_state=settings.random_state)
            sampled = True

        with self._step(trace, "prepare_features", {
            "target": tcol,
            "problem_type": ptype,
            "rows": int(work.shape[0]),
            "chosen_inputs": len(features) if features else "all",
        }) as t:
            X, y, prep_report = data_tools.prepare_features(
                work, tcol, ptype, features,
                numeric_strategy=impute_numeric, categorical_strategy=impute_categorical,
            )
            t.ok(
                f"{prep_report['n_features']} features "
                f"({len(prep_report['numeric_features'])} numeric scaled, "
                f"{len(prep_report['categorical_features'])} categorical encoded); "
                f"dropped: {', '.join(prep_report['dropped_columns'])}"
            )

        if X.shape[1] == 0:
            raise ProcessingError(
                "No usable features left after cleaning — the dataset contains "
                "only unusable columns (constant or very high cardinality)."
            )

        run_info = {
            "n_train_rows_used": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "sampled": sampled,
            "outliers_dropped": outlier_report,
            "split": f"train {int((1 - settings.train_test_ratio) * 100)}% / test {int(settings.train_test_ratio * 100)}%",
            "class_labels": prep_report.get("class_labels"),
        }

        results: list[dict[str, Any]] = []

        if ptype == "clustering":
            run_info.update({"n_train": int(X.shape[0]), "n_test": 0})
            k_lo, k_hi = k_range or K_RANGE
            with ToolTimer(
                trace, "train_model",
                {"model": "cluster model zoo", "k_range": [k_lo, k_hi],
                 "eps_range": list(data_tools.DBSCAN_EPS_VALUES)},
            ) as t:
                best_metrics = None
                best_label = None

                def _consider(name: str, m: dict, model_obj: Any, step) -> None:
                    nonlocal best_metrics, best_label
                    self._save_artifact(name, model_obj)
                    results.append({
                        "name": name,
                        "model_type": "clustering",
                        "status": "ok",
                        "metrics": m,
                        "training_seconds": round(step.duration_s, 3),
                    })
                    if best_metrics is None or m["silhouette"] > best_metrics["silhouette"]:
                        best_metrics, best_label = m, name

                def _fail(name: str, err: Exception, step) -> None:
                    step.fail(str(err))
                    results.append({
                        "name": name,
                        "model_type": "clustering",
                        "status": "failed",
                        "status_message": str(err)[:200],
                        "metrics": {},
                        "training_seconds": 0.0,
                    })

                for k in range(k_lo, k_hi + 1):
                    with self._step(trace, "evaluate_model", {"model": "K-Means", "k": k}) as tt:
                        try:
                            m, km = data_tools.evaluate_clustering(X, k, settings.random_state)
                            step = tt.ok(f"k={k}: silhouette={m['silhouette']:.3f}, sizes={m['cluster_sizes']}")
                            _consider(f"K-Means (k={k})", m, km, step)
                        except Exception as e:  # noqa: BLE001
                            _fail(f"K-Means (k={k})", e, tt)

                for k in range(k_lo, k_hi + 1):
                    with self._step(trace, "evaluate_model", {"model": "Agglomerative", "k": k}) as tt:
                        try:
                            m, agg = data_tools.evaluate_agglomerative(X, k, settings.random_state)
                            step = tt.ok(f"k={k}: silhouette={m['silhouette']:.3f}, sizes={m['cluster_sizes']}")
                            _consider(f"Agglomerative (k={k})", m, agg, step)
                        except Exception as e:  # noqa: BLE001
                            _fail(f"Agglomerative (k={k})", e, tt)

                for eps in data_tools.DBSCAN_EPS_VALUES:
                    name = f"DBSCAN (eps={eps})"
                    with self._step(trace, "evaluate_model", {"model": "DBSCAN", "eps": eps}) as tt:
                        try:
                            m, db = data_tools.evaluate_dbscan(X, eps, 5, settings.random_state)
                            step = tt.ok(
                                f"eps={eps}: {m['k']} clusters, silhouette={m['silhouette']:.3f}, "
                                f"noise={m['noise']}"
                            )
                            _consider(name, m, db, step)
                        except Exception as e:  # noqa: BLE001
                            _fail(name, e, tt)

                if best_metrics:
                    t.ok(f"Best: {best_label} (silhouette={best_metrics['silhouette']:.3f})")
                else:
                    t.fail("All clustering algorithms failed")
            comparison = data_tools.compare_models(results, "clustering")
        else:
            with self._step(trace, "prepare_features", {"purpose": "train/test split"}) as t:
                X_train, X_test, y_train, y_test = data_tools.split_data(
                    X, y, ptype, settings.random_state, settings.train_test_ratio
                )
            run_info.update({"n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0])})
            t.ok(f"Split {int(X_train.shape[0])} train / {int(X_test.shape[0])} test")

            registry = (
                data_tools.REGRESSION_MODELS if ptype == "regression"
                else data_tools.CLASSIFICATION_MODELS
            )
            for model_name in registry:
                model_obj = None
                try:
                    with self._step(trace, "train_model", {"model": model_name}) as tt:
                        model_obj, elapsed = data_tools.train_model(
                            model_name, ptype, X_train, y_train, settings.random_state
                        )
                        tt.ok(f"Trained in {elapsed:.2f}s")
                    self._save_artifact(model_name, model_obj)
                    with self._step(trace, "evaluate_model", {"model": model_name}) as tt:
                        metrics = data_tools.evaluate_model(
                            model_obj, model_name, ptype,
                            X_train, X_test, y_train, y_test,
                            prep_report.get("class_labels"),
                        )
                        primary = metrics.get(metrics.get("primary_metric", "r2"))
                        tt.ok(f"{metrics['primary_metric']}={primary:.4f}")
                    results.append({
                        "name": model_name,
                        "model_type": ptype,
                        "status": "ok",
                        "metrics": metrics,
                        "training_seconds": round(elapsed, 3),
                    })
                except Exception as e:  # noqa: BLE001 — one bad model must not kill the run
                    trace.record("train_model", {"model": model_name}, status="failed", error=str(e)[:300])
                    results.append({
                        "name": model_name,
                        "model_type": ptype,
                        "status": "failed",
                        "status_message": str(e)[:300],
                        "metrics": {},
                        "training_seconds": 0.0,
                    })
            comparison = data_tools.compare_models(results, ptype)

            # Optionally refine the winner. Only the winner: searching all 22
            # would cost more than the rest of the pipeline combined, and a
            # tuned also-ran rarely overtakes a well-fitted leader.
            if tune and ptype != "clustering":
                from .tuner import is_tunable, tune_model

                champion = comparison.get("best")
                if champion and is_tunable(champion):
                    with self._step(trace, "train_model",
                                    {"purpose": "hyperparameter search", "model": champion}) as t:
                        found = tune_model(X_train, y_train, ptype, champion)
                        if found:
                            factory = data_tools.model_factory(ptype, champion)
                            candidate = factory().set_params(**found["params"])
                            candidate.fit(X_train, y_train)
                            metrics = data_tools.evaluate_model(
                                candidate, champion, ptype,
                                X_train, X_test, y_train, y_test,
                                prep_report.get("class_labels"),
                            )
                            entry = next((r for r in results if r["name"] == champion), None)
                            key = metrics.get("primary_metric")
                            before = (entry or {}).get("metrics", {}).get(key)
                            after = metrics.get(key)
                            # Keep the tuned model only if it really is better on
                            # the held-out split, not merely on the search folds.
                            if entry and before is not None and after is not None and after > before:
                                entry["metrics"] = metrics
                                entry["tuned_params"] = found["params"]
                                comparison = data_tools.compare_models(results, ptype)
                                tuning_report = {"model": champion, "params": found["params"],
                                                 "before": before, "after": after, "applied": True}
                                t.ok(f"Tuned {champion}: {key} {before:.4f} -> {after:.4f}")
                            else:
                                tuning_report = {"model": champion, "params": found["params"],
                                                 "before": before, "after": after, "applied": False}
                                t.ok(f"Search found nothing better than the defaults for {champion}")
                            run_info["tuning"] = tuning_report
                        else:
                            t.ok(f"{champion} has no tunable parameters configured")

        # feature importance for the best model (retrain cheaply in-memory is not possible;
        # we already trained — retrieve from a fresh fit on train split for tree/linear models)
        best_name = comparison.get("best")
        best_result = next((r for r in results if r["name"] == best_name), None)
        if best_result and ptype != "clustering":
            try:
                model_obj, _ = data_tools.train_model(
                    self._base_model_name(best_name), ptype, X_train, y_train, settings.random_state
                )
                fi = data_tools.feature_importance(model_obj, X_train)
                best_result["feature_importance"] = fi[:15]
            except Exception:  # noqa: BLE001
                best_result["feature_importance"] = []
        elif best_result:
            fi = self._cluster_centers_importance(X, best_result)
            best_result["feature_importance"] = fi

        for r in results:
            r["rank"] = next(
                (c["rank"] for c in comparison["ranked"] if c["name"] == r["name"]), None
            )
            r["is_best"] = r["name"] == best_name
            r.pop("k", None)

        training = {
            "problem_type": ptype,
            "target": tcol,
            "run_info": run_info,
            "models": results,
            "best_model": best_name,
            "comparison": comparison,
            "clean_report": clean_report,
            "prep_report": prep_report,
            "trace": trace.to_dict(),
            "total_seconds": round(time.perf_counter() - t0, 2),
        }

        with self._step(trace, "compare_models", {"metric": comparison.get("metric_key")}) as t:
            t.ok(
                f"Best model: {best_name} "
                f"({comparison['metric_key']}="
                f"{comparison['ranked'][0]['primary_metric'] if comparison['ranked'] else 'n/a'})"
            )

        with self._step(trace, "generate_report", {"kind": "training-explanation"}) as t:
            local_text = explain_training(training)
            llm_text = None
            if self.llm.enabled:
                llm_text = self.llm.complete(build_interpretation_prompt("training", training))
            training["explanation"] = llm_text or local_text
            training["explanation_source"] = "llm" if llm_text else "local-engine"
            t.ok(f"Training explanation generated (source: {training['explanation_source']})")

        self._save_metadata(ptype, tcol, prep_report, run_info)
        training["trace"] = trace.to_dict()
        return training

    # ------------------------------------------------------------------ #
    # model artifact persistence
    # ------------------------------------------------------------------ #
    def _save_artifact(self, name: str, model: Any) -> None:
        """Persist a trained model as .pkl so users can download & reload it."""
        if self.model_dir is None or model is None:
            return
        try:
            import joblib

            from ..core.security import model_slug

            self.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_dir / f"{model_slug(name)}.pkl")
        except Exception:  # noqa: BLE001 — artifacts must never break the run
            pass

    def _save_metadata(
        self, problem_type: str, target: str | None, prep_report: dict, run_info: dict
    ) -> None:
        if self.model_dir is None:
            return
        try:
            meta = {
                "problem_type": problem_type,
                "target": target,
                "features": prep_report.get("features", []),
                "numeric_features": prep_report.get("numeric_features", []),
                "scaled_columns": prep_report.get("scaled_columns", []),
                "class_labels": prep_report.get("class_labels"),
                "n_train": run_info.get("n_train"),
                "n_test": run_info.get("n_test"),
                "note": (
                    "Numeric features were StandardScaler-normalized before training; "
                    "apply the same preprocessing to new data before calling model.predict()."
                ),
            }
            (self.model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _base_model_name(self, name: str) -> str:
        return name

    def _variance_importance(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        """Features explaining the most variance separate clusters the most."""
        try:
            var = X.var(axis=0).fillna(0).to_numpy()
            total = float(var.sum())
            if total <= 0:
                return []
            order = np.argsort(-var)
            return [
                {"feature": str(X.columns[i]), "importance": round(float(var[i] / total), 4)}
                for i in order[:15]
            ]
        except Exception:  # noqa: BLE001
            return []

    def _cluster_centers_importance(self, X: pd.DataFrame, result: dict) -> list[dict[str, Any]]:
        """'Importance' for clustering: for K-Means, the spread between centers;
        otherwise (Agglomerative/DBSCAN) fall back to variance share."""
        if result["name"].startswith("K-Means"):
            try:
                from sklearn.cluster import KMeans

                k = int(result["metrics"].get("k", 2))
                if k >= 2:
                    model = KMeans(n_clusters=k, n_init=10, random_state=self.settings.random_state)
                    model.fit(X)
                    centers = np.asarray(model.cluster_centers_)
                    spread = centers.max(axis=0) - centers.min(axis=0)
                    total = float(spread.sum())
                    if total > 0:
                        order = np.argsort(-spread)
                        return [
                            {"feature": str(X.columns[i]), "importance": round(float(spread[i] / total), 4)}
                            for i in order[:15]
                        ]
            except Exception:  # noqa: BLE001
                pass
        return self._variance_importance(X)

    def _preview(self, df: pd.DataFrame, n: int | None = None) -> dict[str, Any]:
        n = n or self.settings.preview_rows
        head = df.head(n)
        rows = []
        for _, row in head.iterrows():
            out = []
            for v in row:
                if pd.isna(v):
                    out.append(None)
                elif isinstance(v, (np.integer,)):
                    out.append(int(v))
                elif isinstance(v, (np.floating, float)):
                    f = float(v)
                    out.append(round(f, 6) if abs(f) < 1e9 else round(f, 2))
                elif isinstance(v, (np.bool_, bool)):
                    out.append(bool(v))
                else:
                    out.append(str(v))
            rows.append(out)
        return {
            "columns": [str(c) for c in df.columns],
            "dtypes": [str(df[c].dtype) for c in df.columns],
            "rows": rows,
        }

    def _analysis_plan(
        self, profile: dict[str, Any], missing: dict[str, Any], detection: dict[str, Any]
    ) -> list[str]:
        plan = [
            "1. Profile the dataset (shape, types, duplicates)",
            "2. Detect missing values and plan imputation",
            "3. Compute summary statistics for every column",
            "4. Compute the correlation matrix of numeric columns",
            "5. Detect outliers with the IQR rule",
            "6. Detect the target column and the task type",
            "7. Generate visualizations (histograms, box plot, heatmap, bars, scatter)",
        ]
        if missing["total_missing"] or profile["duplicate_rows"]:
            plan.append("8. Clean the data before training (imputation + dedupe)")
        plan.append(f"Final step: {'train models for ' + detection['problem_type'] + ' and compare them' if detection['problem_type'] != 'clustering' else 'run the cluster model zoo (K-Means, Agglomerative, DBSCAN) and score with silhouette'}")
        return plan
