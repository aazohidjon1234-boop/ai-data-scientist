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
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      setResult(await api.improve(datasetId));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not test the options.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
            Can this score be improved?
          </h3>
          <p className="mt-1 max-w-2xl text-sm text-slate-500 dark:text-slate-400">
            Each option below is actually trained and cross-validated before anything changes, so
            you see a measured score rather than a promise. Nothing is applied until you choose.
          </p>
        </div>
        <Button onClick={run} disabled={running || busy}>
          {running ? "Testing…" : result ? "Test again" : "Test the options"}
        </Button>
      </div>

      {error && <div className="mt-3"><ErrorAlert message={error} onDismiss={() => setError(null)} /></div>}
      {running && (
        <div className="mt-4">
          <Spinner label="Training each option on cross-validation folds — this is real computation…" />
        </div>
      )}

      {result && !running && (
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
                          onClick={() =>
                            onRetrain({
                              target,
                              problem_type: problemType,
                              features: r.changes.features,
                              drop_outliers: r.changes.drop_outliers,
                            })
                          }
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
