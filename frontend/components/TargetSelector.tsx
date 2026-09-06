"use client";

import { useState } from "react";
import type { Analysis, ProblemType } from "@/lib/types";
import { Button, Card } from "./ui";

export default function TargetSelector({
  analysis,
  onTrain,
  busy,
}: {
  analysis: Analysis;
  onTrain: (payload: { target?: string | null; problem_type?: ProblemType | null }) => void;
  busy: boolean;
}) {
  const columns = analysis.profile.columns_info.map((c) => c.name);
  const suggested =
    analysis.target_candidates?.find((c) => c.suggested_task !== "clustering")?.column || null;
  const [target, setTarget] = useState<string>("");
  const [mode, setMode] = useState<"auto" | "clustering">("auto");

  return (
    <Card className="border-indigo-200 bg-indigo-50/40 p-5 dark:border-indigo-500/30 dark:bg-indigo-500/5">
      <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
        {analysis.problem_type === "clustering"
          ? "The agent found no clear target — it will run clustering by default."
          : "Configure the training run"}
      </h3>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        {analysis.problem_type === "clustering"
          ? "Pick a target column to run supervised learning instead, or keep clustering."
          : "Override the auto-detected target or switch to clustering."}
      </p>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <div className="flex overflow-hidden rounded-lg border border-slate-300 dark:border-slate-700">
          {(["auto", "clustering"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1.5 text-sm font-medium transition ${
                mode === m
                  ? "bg-indigo-600 text-white"
                  : "bg-white text-slate-600 hover:bg-slate-50 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800"
              }`}
            >
              {m === "auto" ? "Supervised (pick target)" : "Clustering (no target)"}
            </button>
          ))}
        </div>

        {mode === "auto" && (
          <select
            value={target || (suggested ?? "")}
            onChange={(e) => setTarget(e.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900"
          >
            <option value="">Choose a target column…</option>
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        )}

        <Button
          disabled={busy || (mode === "auto" && !target && !suggested)}
          onClick={() => onTrain(mode === "clustering" ? { problem_type: "clustering" } : { target: target || suggested })}
        >
          {busy ? "Training…" : "Start training"}
        </Button>
      </div>
    </Card>
  );
}
