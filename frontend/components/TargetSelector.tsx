"use client";

import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Analysis, FeatureSuggestion, ProblemType } from "@/lib/types";
import { Badge, Button, Card, Spinner } from "./ui";

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
  datasetId,
  usedFeatures = null,
}: {
  analysis: Analysis;
  onTrain: (payload: TrainPayload) => void;
  busy: boolean;
  hasRun?: boolean;
  datasetId: string;
  /** Columns the latest run actually learned from; null means "all of them". */
  usedFeatures?: string[] | null;
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
  // Start from what the last run used, so a narrowed selection survives the
  // retrain instead of silently resetting to every column.
  const [features, setFeatures] = useState<string[]>(usedFeatures ?? columns);
  const [touched, setTouched] = useState(false);

  // A new run may have changed the columns; adopt them unless the user is
  // mid-edit, in which case their choice wins.
  useEffect(() => {
    if (touched) return;
    setFeatures(usedFeatures ?? columns);
  }, [usedFeatures, columns, touched]);

  // The target can never also be an input — dropping it here keeps the count
  // honest instead of silently discarding it at training time.
  useEffect(() => {
    setFeatures((current) => current.filter((c) => c !== target));
  }, [target]);

  const activeTarget = mode === "clustering" ? "" : target;
  const selectable = columns.filter((c) => c !== activeTarget);
  const chosen = features.filter((c) => c !== activeTarget);
  const allChosen = chosen.length === selectable.length;

  const toggle = (name: string) => {
    setTouched(true);
    setFeatures((current) =>
      current.includes(name) ? current.filter((c) => c !== name) : [...current, name],
    );
  };

  const [advice, setAdvice] = useState<FeatureSuggestion | null>(null);
  const [advising, setAdvising] = useState(false);
  const [adviceError, setAdviceError] = useState<string | null>(null);

  // Suggestions are relative to a target, so they are invalidated when it moves.
  useEffect(() => {
    setAdvice(null);
    setAdviceError(null);
  }, [target, mode]);

  const askForAdvice = async () => {
    setAdvising(true);
    setAdviceError(null);
    try {
      // Problem type is left to the backend, which reads it from the analysis.
      const result = await api.suggestFeatures(datasetId, target || suggested, null);
      setAdvice(result);
      setTouched(true);
      setFeatures(result.recommended);
    } catch (e) {
      setAdviceError(e instanceof ApiError ? e.message : "Could not analyse the columns.");
    } finally {
      setAdvising(false);
    }
  };

  const verdictOf = (name: string) => advice?.columns.find((c) => c.column === name);

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
            {mode === "auto" && (
              <button
                onClick={askForAdvice}
                disabled={advising || (!target && !suggested)}
                className="rounded-md border border-indigo-400 bg-indigo-600 px-2.5 py-1 text-xs font-medium text-white transition hover:bg-indigo-700 disabled:opacity-40"
              >
                {advising ? "Analysing…" : "✨ Suggest columns"}
              </button>
            )}
            <button
              onClick={() => { setTouched(true); setFeatures(selectable); }}
              disabled={allChosen}
              className="rounded-md border border-slate-300 px-2.5 py-1 text-xs text-slate-600 transition hover:border-indigo-400 hover:text-indigo-600 disabled:opacity-40 dark:border-slate-700 dark:text-slate-300"
            >
              Select all
            </button>
            <button
              onClick={() => { setTouched(true); setFeatures([]); }}
              disabled={chosen.length === 0}
              className="rounded-md border border-slate-300 px-2.5 py-1 text-xs text-slate-600 transition hover:border-indigo-400 hover:text-indigo-600 disabled:opacity-40 dark:border-slate-700 dark:text-slate-300"
            >
              Clear
            </button>
          </div>
        </div>

        {advising && (
          <div className="mt-2">
            <Spinner label="Scoring each column against the target and cross-validating…" />
          </div>
        )}
        {adviceError && (
          <p className="mt-2 text-sm text-rose-600 dark:text-rose-400">{adviceError}</p>
        )}
        {advice && !advising && (
          <div className="mt-2 rounded-lg border border-indigo-200 bg-white p-3 text-sm dark:border-indigo-500/30 dark:bg-slate-900">
            <p className="text-slate-800 dark:text-slate-200">{advice.summary}</p>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Scored with mutual information, then checked by {advice.evaluation.folds}-fold
              cross-validation on {advice.evaluation.rows_used.toLocaleString()} rows using a{" "}
              {advice.evaluation.model}. The {advice.recommended.length} recommended column
              {advice.recommended.length === 1 ? " is" : "s are"} ticked below — change anything you
              disagree with.
            </p>
          </div>
        )}

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
                  {(() => {
                    const v = verdictOf(name);
                    if (!v) return null;
                    const tone =
                      v.verdict === "keep" ? "green" : v.verdict === "drop" ? "red" : "amber";
                    return (
                      <span title={v.reason}>
                        <Badge tone={tone}>{v.verdict}</Badge>
                      </span>
                    );
                  })()}
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
