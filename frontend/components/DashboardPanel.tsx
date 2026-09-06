"use client";

import { useCallback, useEffect, useState } from "react";

import PlotlyChart from "@/components/PlotlyChart";
import { Badge, Button, Card, EmptyState, ErrorAlert, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import type { DashboardResult } from "@/lib/types";

type Selection = Record<string, string[]>;

function formatValue(value: number | string, format: string): string {
  if (typeof value === "string") return value;
  if (!Number.isFinite(value)) return "—";
  if (format === "int") return value.toLocaleString();
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 2 })}M`;
  if (abs >= 10_000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export default function DashboardPanel({ datasetId }: { datasetId: string }) {
  const [data, setData] = useState<DashboardResult | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selection, setSelection] = useState<Selection>({});
  const [measure, setMeasure] = useState<string | null>(null);
  const [dimension, setDimension] = useState<string | null>(null);

  const load = useCallback(
    async (sel: Selection, m: string | null, d: string | null) => {
      setBusy(true);
      setError(null);
      try {
        const filters = Object.entries(sel)
          .filter(([, values]) => values.length > 0)
          .map(([column, values]) => ({ column, op: "in", value: values }));
        const result = await api.dashboard(datasetId, {
          filters,
          measure: m,
          dimension: d,
        });
        setData(result);
        // Adopt whatever the backend picked, so the selectors show the truth.
        setMeasure(result.measure);
        setDimension(result.dimension);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Could not build the dashboard.");
      } finally {
        setBusy(false);
      }
    },
    [datasetId],
  );

  useEffect(() => {
    load({}, null, null);
  }, [load]);

  const toggleValue = (column: string, value: string) => {
    const current = selection[column] ?? [];
    const next = current.includes(value)
      ? current.filter((v) => v !== value)
      : [...current, value];
    const updated = { ...selection, [column]: next };
    setSelection(updated);
    load(updated, measure, dimension);
  };

  const clearFilters = () => {
    setSelection({});
    load({}, measure, dimension);
  };

  const activeCount = Object.values(selection).reduce((n, v) => n + v.length, 0);

  if (!data && busy) return <Spinner label="Composing the dashboard…" />;
  if (error && !data) return <ErrorAlert message={error} />;
  if (!data) return null;

  const { layout } = data;

  return (
    <div className="space-y-4">
      {/* ------------------------------------------------------- controls */}
      <Card className="p-4">
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-slate-500 dark:text-slate-400">
            Measure
            <select
              value={measure ?? ""}
              onChange={(e) => {
                const v = e.target.value || null;
                setMeasure(v);
                load(selection, v, dimension);
              }}
              className="mt-1 block rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950 dark:text-white"
            >
              <option value="">(row counts)</option>
              {layout.measures.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </label>

          <label className="text-xs text-slate-500 dark:text-slate-400">
            Break down by
            <select
              value={dimension ?? ""}
              onChange={(e) => {
                const v = e.target.value || null;
                setDimension(v);
                load(selection, measure, v);
              }}
              className="mt-1 block rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950 dark:text-white"
            >
              <option value="">(none)</option>
              {layout.dimensions.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </label>

          <div className="ml-auto flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
            {busy && <Spinner />}
            <span>
              {data.rows_shown.toLocaleString()} of {data.rows_total.toLocaleString()} rows
            </span>
            {activeCount > 0 && (
              <button
                onClick={clearFilters}
                className="rounded-md border border-slate-300 px-2.5 py-1 transition hover:border-indigo-400 hover:text-indigo-600 dark:border-slate-700"
              >
                Clear {activeCount} filter{activeCount === 1 ? "" : "s"}
              </button>
            )}
          </div>
        </div>

        {/* ------------------------------------------------------ filters */}
        {data.filter_options.length > 0 && (
          <div className="mt-3 space-y-2 border-t border-slate-200 pt-3 dark:border-slate-800">
            {data.filter_options.map((option) => (
              <div key={option.column} className="flex flex-wrap items-center gap-1.5">
                <span className="w-28 shrink-0 truncate text-xs font-medium text-slate-600 dark:text-slate-300">
                  {option.column}
                </span>
                {option.values.map((value) => {
                  const on = (selection[option.column] ?? []).includes(value);
                  return (
                    <button
                      key={value}
                      onClick={() => toggleValue(option.column, value)}
                      className={`rounded-full border px-2.5 py-0.5 text-xs transition ${
                        on
                          ? "border-indigo-500 bg-indigo-600 text-white"
                          : "border-slate-300 text-slate-600 hover:border-indigo-400 dark:border-slate-700 dark:text-slate-300"
                      }`}
                    >
                      {value}
                    </button>
                  );
                })}
                {option.truncated && (
                  <span className="text-xs text-slate-400">+ more</span>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {data.empty ? (
        <EmptyState
          title="No rows match these filters"
          subtitle="Remove a filter to bring data back."
          action={<Button onClick={clearFilters}>Clear filters</Button>}
        />
      ) : (
        <>
          {/* --------------------------------------------------------- KPIs */}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
            {data.kpis.map((kpi) => (
              <Card key={kpi.label} className="p-4">
                <div className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  {kpi.label}
                </div>
                <div
                  className="mt-1 truncate text-2xl font-semibold text-slate-900 dark:text-white"
                  title={String(kpi.value)}
                >
                  {formatValue(kpi.value, kpi.format)}
                </div>
                <div className="mt-0.5 truncate text-xs text-slate-500 dark:text-slate-400">
                  {kpi.hint}
                </div>
              </Card>
            ))}
          </div>

          {/* ------------------------------------------------------- charts */}
          <div className="grid gap-4 lg:grid-cols-2">
            {data.charts.map((chart) => (
              <Card key={chart.id} className="p-4">
                <h3 className="mb-1 text-sm font-semibold text-slate-700 dark:text-slate-200">
                  {chart.title}
                </h3>
                <PlotlyChart figure={chart} />
              </Card>
            ))}
          </div>

          <p className="text-xs text-slate-400">
            Every tile and chart is recomputed with pandas from the filtered rows.{" "}
            {data.date_column && <Badge tone="cyan">timeline: {data.date_column}</Badge>}
          </p>
        </>
      )}
    </div>
  );
}
