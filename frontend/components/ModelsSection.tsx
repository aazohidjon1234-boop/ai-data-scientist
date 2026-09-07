"use client";

import { useMemo } from "react";
import type { ModelRun, ProblemType } from "@/lib/types";
import { fmtNum, metricLabel } from "@/lib/format";
import { Badge, Card } from "./ui";
import PlotlyChart from "./PlotlyChart";

const METRICS_ORDER: Record<ProblemType, string[]> = {
  regression: ["r2", "mae", "rmse", "mse"],
  classification: ["f1", "accuracy", "precision", "recall"],
  clustering: ["silhouette", "inertia"],
};

function buildComparisonFigure(run: ModelRun) {
  const metrics = METRICS_ORDER[run.problem_type];
  const ok = run.models.filter((m) => m.status === "ok");
  const traces = metrics
    .filter((mk) => ok.some((m) => m.metrics[mk] !== undefined))
    .map((mk, idx) => ({
      type: "bar",
      name: metricLabel(mk),
      x: ok.map((m) => m.name),
      y: ok.map((m) => (m.metrics[mk] as number) ?? 0),
      marker: {
        color: ok.map((m) => (m.is_best ? "#6366f1" : idx === 0 ? "#94a3b8" : "#cbd5e1")),
      },
      opacity: ok.map((m) => (m.is_best ? 1 : 0.75)),
    }));
  return {
    id: "comparison",
    kind: "comparison",
    title: "Model comparison",
    data: { data: traces, layout: { barmode: "group" } },
  };
}

export default function ModelsSection({ run }: { run: ModelRun }) {
  const figure = useMemo(() => buildComparisonFigure(run), [run]);
  const metrics = METRICS_ORDER[run.problem_type];
  const best = run.models.find((m) => m.is_best);

  return (
    <div className="space-y-4">
      {run.models.some((m) => m.status !== "ok") && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-2.5 text-sm text-amber-800 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200">
          {run.models.filter((m) => m.status !== "ok").length} model(s) failed to train — details in
          the table below.
        </div>
      )}

      <p className="text-xs text-slate-500 dark:text-slate-400">
        {run.ranked_by === "cross_validation"
          ? "Ranked by 5-fold cross-validation — on data this size a single hold-out split can hand first place to whichever model got the kinder split."
          : "Ranked by the held-out test split."}
      </p>

      <Card>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-slate-50 text-left dark:bg-slate-800">
                <th className="px-3 py-2 font-semibold text-slate-600 dark:text-slate-300">Model</th>
                {metrics.map((mk) => (
                  <th key={mk} className="whitespace-nowrap px-3 py-2 text-right font-semibold text-slate-600 dark:text-slate-300">
                    {metricLabel(mk)}
                  </th>
                ))}
                <th
                  className="px-3 py-2 text-right font-semibold text-slate-600 dark:text-slate-300"
                  title="Cross-validated score — used for ranking when the data is small enough that one split is unreliable"
                >
                  CV
                </th>
                <th className="px-3 py-2 text-right font-semibold text-slate-600 dark:text-slate-300">Time (s)</th>
                <th className="px-3 py-2 font-semibold text-slate-600 dark:text-slate-300">Status</th>
              </tr>
            </thead>
            <tbody>
              {[...run.models]
                // Rank first; fall back to the primary metric so the best model
                // still leads even if ranking data is missing. Failed runs sink.
                .sort((a, b) => {
                  if (a.rank != null && b.rank != null) return a.rank - b.rank;
                  if (a.rank != null) return -1;
                  if (b.rank != null) return 1;
                  const av = a.status === "ok" ? a.primary_metric ?? -Infinity : -Infinity;
                  const bv = b.status === "ok" ? b.primary_metric ?? -Infinity : -Infinity;
                  return bv - av;
                })
                .map((m) => (
                  <tr
                    key={m.name}
                    className={`border-t border-slate-100 dark:border-slate-800 ${
                      m.is_best ? "bg-indigo-50/60 dark:bg-indigo-500/10" : ""
                    }`}
                  >
                    <td className="whitespace-nowrap px-3 py-2 font-medium text-slate-800 dark:text-slate-100">
                      {m.is_best && <span className="mr-1.5">⭐</span>}
                      {m.name}
                      {m.download_url && (
                        <a
                          href={m.download_url}
                          download
                          title="Download the trained model (.pkl — load with joblib.load)"
                          className="ml-2 text-indigo-500 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300"
                        >
                          ⬇ pkl
                        </a>
                      )}
                    </td>
                    {metrics.map((mk) => (
                      <td key={mk} className="whitespace-nowrap px-3 py-2 text-right tabular-nums text-slate-600 dark:text-slate-300">
                        {m.metrics[mk] !== undefined ? fmtNum(m.metrics[mk]) : "—"}
                      </td>
                    ))}
                    <td className="whitespace-nowrap px-3 py-2 text-right tabular-nums text-slate-600 dark:text-slate-300">
                      {m.cv_score != null ? (
                        <>
                          {fmtNum(m.cv_score)}
                          {m.cv_std != null && (
                            <span className="ml-1 text-xs text-slate-400">±{fmtNum(m.cv_std)}</span>
                          )}
                        </>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-slate-500">{fmtNum(m.training_seconds)}</td>
                    <td className="px-3 py-2">
                      {m.status === "ok" ? (
                        <Badge tone="green">trained</Badge>
                      ) : (
                        <Badge tone="red">
                          failed{m.status_message ? ` — ${m.status_message.slice(0, 60)}` : ""}
                        </Badge>
                      )}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-4">
          <h3 className="mb-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
            Metrics by model
          </h3>
          <PlotlyChart figure={figure} height={300} />
        </Card>

        <Card className="p-4">
          <h3 className="mb-3 text-sm font-semibold text-slate-700 dark:text-slate-200">
            {run.problem_type === "clustering"
              ? "Feature separation (best k)"
              : `Most important features — ${best?.name ?? "best model"}`}
          </h3>
          {best && best.feature_importance && best.feature_importance.length > 0 ? (
            <div className="space-y-2">
              {best.feature_importance.slice(0, 10).map((f, i) => {
                const max = best.feature_importance![0]?.importance || 1;
                return (
                  <div key={f.feature}>
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-700 dark:text-slate-200">{f.feature}</span>
                      <span className="tabular-nums text-slate-400">{fmtNum(f.importance)}</span>
                    </div>
                    <div className="mt-1 h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
                      <div
                        className="h-1.5 rounded-full bg-indigo-500"
                        style={{ width: `${Math.max(3, (f.importance / max) * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-slate-500">Feature importances are not available for this model.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
