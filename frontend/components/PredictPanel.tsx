"use client";

import { useEffect, useState } from "react";

import { Badge, Button, Card, ErrorAlert, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import type { PredictResult, PredictSchema } from "@/lib/types";

export default function PredictPanel({ datasetId }: { datasetId: string }) {
  const [schema, setSchema] = useState<PredictSchema | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .predictSchema(datasetId)
      .then((s) => {
        setSchema(s);
        // Prefill from a real training row so the form is usable immediately.
        setValues(
          Object.fromEntries(
            s.required_columns.map((c) => [c, String(s.example_row?.[c] ?? "")]),
          ),
        );
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : "Prediction is unavailable."));
  }, [datasetId]);

  const run = async () => {
    if (!schema) return;
    setBusy(true);
    setError(null);
    try {
      // Send numbers as numbers; a quoted number would be encoded as a category.
      const row = Object.fromEntries(
        Object.entries(values).map(([k, v]) => {
          const trimmed = v.trim();
          const asNumber = Number(trimmed);
          return [k, trimmed !== "" && Number.isFinite(asNumber) ? asNumber : trimmed];
        }),
      );
      setResult(await api.predict(datasetId, [row]));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not predict.");
    } finally {
      setBusy(false);
    }
  };

  if (error && !schema) return <ErrorAlert message={error} />;
  if (!schema) return <Spinner label="Loading the model's inputs…" />;

  const top = result?.confidence?.[0]
    ? Object.entries(result.confidence[0]).sort((a, b) => b[1] - a[1])[0]
    : null;

  return (
    <Card className="p-5">
      <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
        Predict {schema.target ?? "the target"} for a new row
      </h3>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        Uses <strong>{schema.model}</strong> — the model this run selected — with the exact scaling
        and encoding it was trained on.
      </p>

      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {schema.required_columns.map((column) => (
          <label key={column} className="text-xs text-slate-500 dark:text-slate-400">
            {column}
            <input
              value={values[column] ?? ""}
              onChange={(e) => setValues((v) => ({ ...v, [column]: e.target.value }))}
              className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-white"
            />
          </label>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button onClick={run} disabled={busy}>
          {busy ? "Predicting…" : "Predict"}
        </Button>
        {result && (
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="text-slate-500 dark:text-slate-400">Prediction:</span>
            <Badge tone="indigo">{String(result.predictions[0])}</Badge>
            {top && (
              <span className="text-xs text-slate-500 dark:text-slate-400">
                {(top[1] * 100).toFixed(1)}% confident
              </span>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="mt-3">
          <ErrorAlert message={error} onDismiss={() => setError(null)} />
        </div>
      )}
    </Card>
  );
}
