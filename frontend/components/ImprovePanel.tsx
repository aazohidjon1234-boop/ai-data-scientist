"use client";

import { useState } from "react";

import { Badge, Button, Card, ErrorAlert, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import type { ImproveResult, ProblemType } from "@/lib/types";

type TrainPayload = {
  target?: string | null;
  problem_type?: ProblemType | null;
  features?: string[] | null;
  drop_outliers?: boolean;
  tune?: boolean;
  impute_numeric?: string;
  impute_categorical?: string;
  engineered?: Record<string, unknown>[] | null;
};

export default function ImprovePanel({
  datasetId,
  target,
  problemType,
  onRetrain,
  busy,
}: {
  datasetId: string;
  target: string | null;
  problemType: ProblemType | null;
  onRetrain: (payload: TrainPayload) => void;
  busy: boolean;
}) {
  const [result, setResult] = useState<ImproveResult | null>(null);
  const [running, setRunning] = useState(false);
  const [autoRunning, setAutoRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const test = async (): Promise<ImproveResult | null> => {
    setError(null);
    setNote(null);
    try {
      const r = await api.improve(datasetId);
      setResult(r);
      return r;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not test the options.");
      return null;
    }
  };

  const run = async () => {
    setRunning(true);
    await test();
    setRunning(false);
  };

  const applyRecipe = (r: ImproveResult["recipes"][number]) =>
    onRetrain({
      target,
      problem_type: problemType,
      features: r.changes.features,
      drop_outliers: r.changes.drop_outliers,
      tune: Boolean(r.changes.tune),
      impute_numeric: r.changes.impute_numeric,
      impute_categorical: r.changes.impute_categorical,
      engineered: r.changes.engineered ?? null,
    });

  /** Test everything and retrain on the winner — the whole loop in one click. */
  const improveAutomatically = async () => {
    setAutoRunning(true);
    const r = await test();
    setAutoRunning(false);
    if (!r) return;
    const winner = r.recipes.find((x) => x.key === r.recommended);
    if (!winner || r.recommended === "baseline") {
      setNote(
        "Tested every option — none beat the current setup by a meaningful margin, so nothing was changed.",
      );
      return;
    }
    setNote(`Applying "${winner.label}" and retraining…`);
    applyRecipe(winner);
  };

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
            Can this score be improved?
          </h3>
          <p className="mt-1 max-w-2xl text-sm text-slate-500 dark:text-slate-400">
            <strong>Improve automatically</strong> tests every option and retrains on the winner —
            or tells you plainly that nothing helped. Use <strong>Test the options</strong> to see
            the scores first and choose yourself. Either way the options are really trained and
            cross-validated, so the numbers are measured rather than promised.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={improveAutomatically} disabled={running || autoRunning || busy}>
            {autoRunning ? "Testing…" : "⚡ Improve automatically"}
          </Button>
          <Button variant="secondary" onClick={run} disabled={running || autoRunning || busy}>
            {running ? "Testing…" : result ? "Test again" : "Test the options"}
          </Button>
        </div>
      </div>

      {error && <div className="mt-3"><ErrorAlert message={error} onDismiss={() => setError(null)} /></div>}
      {note && (
        <p className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-700 dark:bg-slate-800/60 dark:text-slate-200">
          {note}
        </p>
      )}
      {(running || autoRunning) && (
        <div className="mt-4">
          <Spinner label="Training each option on cross-validation folds — this is real computation…" />
        </div>
      )}

      {result && !running && !autoRunning && (
        <div className="mt-4 space-y-3">
          <p
            className="text-sm text-slate-800 dark:text-slate-200"
            dangerouslySetInnerHTML={{
              __html: result.summary.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>"),
            }}
          />

          <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left dark:bg-slate-900/60">
                <tr>
                  <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-300">Option</th>
                  <th className="px-3 py-2 text-right font-medium text-slate-600 dark:text-slate-300">
                    {result.metric}
                  </th>
                  <th className="px-3 py-2 text-right font-medium text-slate-600 dark:text-slate-300">
                    Change
                  </th>
                  <th className="px-3 py-2 font-medium text-slate-600 dark:text-slate-300">Rows</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {result.recipes.map((r) => {
                  const winner = r.key === result.recommended && r.key !== "baseline";
                  return (
                    <tr
                      key={r.key}
                      className={`border-t border-slate-100 dark:border-slate-800 ${
                        winner ? "bg-green-50/60 dark:bg-green-500/10" : ""
                      }`}
                    >
                      <td className="px-3 py-2">
                        <div className="font-medium text-slate-800 dark:text-slate-100">
                          {winner && <span className="mr-1">✅</span>}
                          {r.label}
                        </div>
                        <div className="text-xs text-slate-500 dark:text-slate-400">{r.why}</div>
                        {r.params && (
                          <div className="mt-0.5 font-mono text-[11px] text-slate-400">
                            {Object.entries(r.params)
                              .map(([k, v]) => `${k}=${v}`)
                              .join("  ")}
                          </div>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-slate-800 dark:text-slate-200">
                        {r.score === null ? "—" : r.score.toFixed(4)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {r.delta === null || r.key === "baseline" ? (
                          <span className="text-slate-400">—</span>
                        ) : (
                          <span
                            className={
                              r.delta > 0
                                ? "text-green-600 dark:text-green-400"
                                : r.delta < 0
                                  ? "text-rose-600 dark:text-rose-400"
                                  : "text-slate-400"
                            }
                          >
                            {r.delta > 0 ? "+" : ""}
                            {r.delta.toFixed(4)}
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-xs text-slate-500 dark:text-slate-400">
                        {r.rows_used.toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <button
                          disabled={busy}
                          onClick={() => applyRecipe(r)}
                          className="whitespace-nowrap rounded-md border border-indigo-400 px-2.5 py-1 text-xs font-medium text-indigo-600 transition hover:bg-indigo-600 hover:text-white disabled:opacity-40 dark:text-indigo-300"
                        >
                          Apply &amp; retrain
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
            <Badge tone="slate">
              {result.folds}-fold cross-validation
            </Badge>
            <Badge tone="amber">
              {result.outliers.rows_flagged} outlier rows ({result.outliers.pct}%)
            </Badge>
            {result.imputation && (
              <Badge tone="slate">
                filling gaps: {result.imputation.numeric} / {result.imputation.categorical}
                {result.imputation.columns_with_gaps.length === 0 && " (nothing missing)"}
              </Badge>
            )}
            {result.outliers.skipped_reason && (
              <span>Outlier removal skipped: {result.outliers.skipped_reason}.</span>
            )}
            <span>
              Scores come from the same kind of models the full run uses, so an improvement here
              should survive the whole model zoo.
            </span>
          </div>
        </div>
      )}
    </Card>
  );
}
