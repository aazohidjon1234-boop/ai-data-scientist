"use client";

import { useEffect, useMemo, useState } from "react";
import type { Analysis, ProblemType } from "@/lib/types";
import { Badge, Button, Card } from "./ui";

type TrainPayload = {
  target?: string | null;
  problem_type?: ProblemType | null;
  features?: string[] | null;
};

export default function TargetSelector({
  analysis,
  onTrain,
  busy,
  hasRun = false,
}: {
  analysis: Analysis;
  onTrain: (payload: TrainPayload) => void;
  busy: boolean;
  hasRun?: boolean;
}) {
  const info = analysis.profile.columns_info;
  const columns = useMemo(() => info.map((c) => c.name), [info]);
  const rows = analysis.profile.rows || 1;

  const suggested =
    analysis.target_candidates?.find((c) => c.suggested_task !== "clustering")?.column || null;

  const [mode, setMode] = useState<"auto" | "clustering">(
    analysis.problem_type === "clustering" ? "clustering" : "auto",
  );
  const [target, setTarget] = useState<string>(suggested ?? "");
  // Everything except the target, which is the behaviour before this existed.
  const [features, setFeatures] = useState<string[]>(columns);

  // The target can never also be an input — dropping it here keeps the count
  // honest instead of silently discarding it at training time.
  useEffect(() => {
    setFeatures((current) => current.filter((c) => c !== target));
  }, [target]);

  const activeTarget = mode === "clustering" ? "" : target;
  const selectable = columns.filter((c) => c !== activeTarget);
  const chosen = features.filter((c) => c !== activeTarget);
  const allChosen = chosen.length === selectable.length;

  const toggle = (name: string) =>
    setFeatures((current) =>
      current.includes(name) ? current.filter((c) => c !== name) : [...current, name],
    );

  const meta = (name: string) => info.find((c) => c.name === name);
  const missingPct = (name: string) => {
    const m = meta(name);
    return m ? Math.round((m.missing / rows) * 100) : 0;
  };

  const canTrain =
    !busy && chosen.length > 0 && (mode === "clustering" || Boolean(target || suggested));

  return (
    <Card className="border-indigo-200 bg-indigo-50/40 p-5 dark:border-indigo-500/30 dark:bg-indigo-500/5">
      <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
        {hasRun ? "Train again with different columns" : "Configure the training run"}
      </h3>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        Choose what to predict and which columns the models may learn from.
        {hasRun && " Running again replaces the results above."}
      </p>

      {/* ---------------------------------------------------------- target */}
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
          <label className="text-sm text-slate-600 dark:text-slate-300">
            Target (y):{" "}
            <select
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900"
            >
              <option value="">Choose a target column…</option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                  {c === suggested ? "  (suggested)" : ""}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {/* -------------------------------------------------------- features */}
      <div className="mt-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-sm font-medium text-slate-700 dark:text-slate-200">
            Input columns (X){" "}
            <span className="font-normal text-slate-500 dark:text-slate-400">
              — {chosen.length} of {selectable.length} selected
            </span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setFeatures(selectable)}
              disabled={allChosen}
              className="rounded-md border border-slate-300 px-2.5 py-1 text-xs text-slate-600 transition hover:border-indigo-400 hover:text-indigo-600 disabled:opacity-40 dark:border-slate-700 dark:text-slate-300"
            >
              Select all
            </button>
            <button
              onClick={() => setFeatures([])}
              disabled={chosen.length === 0}
              className="rounded-md border border-slate-300 px-2.5 py-1 text-xs text-slate-600 transition hover:border-indigo-400 hover:text-indigo-600 disabled:opacity-40 dark:border-slate-700 dark:text-slate-300"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="mt-2 grid max-h-64 gap-1 overflow-y-auto rounded-lg border border-slate-200 bg-white p-2 dark:border-slate-800 dark:bg-slate-900 sm:grid-cols-2">
          {selectable.map((name) => {
            const m = meta(name);
            const pct = missingPct(name);
            return (
              <label
                key={name}
                className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                <input
                  type="checkbox"
                  checked={features.includes(name)}
                  onChange={() => toggle(name)}
                  className="h-4 w-4 accent-indigo-600"
                />
                <span className="truncate text-slate-800 dark:text-slate-200" title={name}>
                  {name}
                </span>
                <span className="ml-auto flex shrink-0 items-center gap-1">
                  <Badge tone={m?.kind === "numeric" ? "indigo" : "slate"}>
                    {m?.kind ?? "?"}
                  </Badge>
                  {pct >= 20 && <Badge tone="amber">{pct}% empty</Badge>}
                </span>
              </label>
            );
          })}
        </div>

        {chosen.length === 0 && (
          <p className="mt-2 text-sm text-rose-600 dark:text-rose-400">
            Select at least one input column.
          </p>
        )}
        {mode === "auto" && !target && !suggested && (
          <p className="mt-2 text-sm text-rose-600 dark:text-rose-400">
            Choose a target column to predict.
          </p>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button
          disabled={!canTrain}
          onClick={() =>
            onTrain(
              mode === "clustering"
                ? { problem_type: "clustering", features: chosen }
                : { target: target || suggested, features: chosen },
            )
          }
        >
          {busy ? "Training…" : "Start training"}
        </Button>
        <span className="text-xs text-slate-500 dark:text-slate-400">
          {mode === "clustering"
            ? `Clustering on ${chosen.length} column${chosen.length === 1 ? "" : "s"}.`
            : `Predicting ${target || suggested || "…"} from ${chosen.length} column${
                chosen.length === 1 ? "" : "s"
              }.`}
        </span>
      </div>
    </Card>
  );
}
