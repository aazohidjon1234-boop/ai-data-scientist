"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Dataset } from "@/lib/types";
import { taskLabel, timeAgo } from "@/lib/format";
import UploadCard from "@/components/UploadCard";
import { Badge, Card, EmptyState, ErrorAlert, Spinner } from "@/components/ui";

const TASK_TONE: Record<string, "indigo" | "green" | "cyan"> = {
  Regression: "indigo",
  Classification: "green",
  Clustering: "cyan",
};

export default function DashboardPage() {
  const [datasets, setDatasets] = useState<Dataset[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    api
      .listDatasets()
      .then(setDatasets)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load datasets."));
  };

  useEffect(refresh, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="mt-1 max-w-2xl text-sm text-slate-500 dark:text-slate-400">
          Upload a CSV and the AI Data Scientist Agent profiles the data, detects the problem,
          cleans it, trains and compares real machine-learning models, then explains the results in
          plain language.
        </p>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      <Card className="p-5 sm:p-6">
        <div className="grid gap-8 lg:grid-cols-[1.2fr_1fr]">
          <div>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
              Upload a dataset
            </h2>
            <UploadCard onDataset={(ds) => (window.location.href = `/analysis/${ds.id}`)} />
          </div>

          <div className="hidden lg:block">
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
              How it works
            </h2>
            <ol className="space-y-3">
              {[
                ["Inspect", "Rows, columns, types, missing values, duplicates, correlations."],
                ["Decide", "Detects the target and whether it's regression, classification or clustering."],
                ["Model", "Trains several scikit-learn models on a train/test split and compares real metrics."],
                ["Explain", "The agent explains what happened, which features matter and how to improve — no invented numbers."],
              ].map(([t, d], i) => (
                <li key={t} className="flex gap-3">
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white">
                    {i + 1}
                  </span>
                  <div>
                    <div className="text-sm font-semibold text-slate-800 dark:text-slate-100">{t}</div>
                    <div className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">{d}</div>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </Card>

      <div id="analyses">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500">
            Recent analyses
          </h2>
          {datasets && datasets.length > 0 && (
            <button onClick={refresh} className="text-xs font-medium text-indigo-600 hover:underline dark:text-indigo-400">
              Refresh
            </button>
          )}
        </div>

        {!datasets ? (
          <Card className="p-8">
            <Spinner label="Loading datasets…" />
          </Card>
        ) : datasets.length === 0 ? (
          <EmptyState
            title="No datasets yet"
            subtitle="Upload a CSV above or load one of the example datasets to see the agent in action."
          />
        ) : (
          <Card>
            <div className="thin-scroll overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="bg-slate-50 text-left dark:bg-slate-800">
                    {["Dataset", "Size", "Task", "Target", "Best model", "Status", "Created", ""].map((h) => (
                      <th key={h} className="whitespace-nowrap px-4 py-2.5 font-semibold text-slate-600 dark:text-slate-300">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((d) => {
                    const task = taskLabel(d.problem_type);
                    return (
                      <tr
                        key={d.id}
                        className="border-t border-slate-100 hover:bg-slate-50/70 dark:border-slate-800 dark:hover:bg-slate-800/40"
                      >
                        <td className="px-4 py-2.5">
                          <div className="font-medium text-slate-800 dark:text-slate-100">{d.name}</div>
                          <div className="text-xs text-slate-400">{d.original_filename}</div>
                        </td>
                        <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-slate-600 dark:text-slate-300">
                          {d.rows.toLocaleString()} × {d.columns}
                        </td>
                        <td className="px-4 py-2.5">
                          {d.problem_type ? <Badge tone={TASK_TONE[task] || "slate"}>{task}</Badge> : <span className="text-slate-400">—</span>}
                        </td>
                        <td className="whitespace-nowrap px-4 py-2.5 font-mono text-xs text-slate-500">{d.target || "—"}</td>
                        <td className="whitespace-nowrap px-4 py-2.5 text-slate-600 dark:text-slate-300">{d.best_model || "—"}</td>
                        <td className="px-4 py-2.5">
                          {d.status === "trained" ? (
                            <Badge tone="green">trained</Badge>
                          ) : d.status === "analyzed" ? (
                            <Badge tone="indigo">analyzed</Badge>
                          ) : (
                            <Badge>uploaded</Badge>
                          )}
                        </td>
                        <td className="whitespace-nowrap px-4 py-2.5 text-slate-500">{timeAgo(d.created_at)}</td>
                        <td className="px-4 py-2.5 text-right">
                          <Link
                            href={`/analysis/${d.id}`}
                            className="font-medium text-indigo-600 hover:underline dark:text-indigo-400"
                          >
                            Open →
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
