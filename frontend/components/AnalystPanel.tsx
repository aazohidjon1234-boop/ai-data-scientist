"use client";

import { useCallback, useEffect, useState } from "react";

import DashboardPanel from "@/components/DashboardPanel";
import PlotlyChart from "@/components/PlotlyChart";
import { Badge, Button, Card, EmptyState, ErrorAlert, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import type {
  AskResult,
  DatasetSchema,
  Insight,
  InsightsResult,
  QueryTable,
  SegmentResult,
  TrendResult,
} from "@/lib/types";

type Tab = "dashboard" | "ask" | "insights" | "compare" | "trend";

const TABS: { key: Tab; label: string; hint: string }[] = [
  { key: "dashboard", label: "Dashboard", hint: "KPIs and charts with filters" },
  { key: "ask", label: "Ask", hint: "Question the data directly" },
  { key: "insights", label: "Insights", hint: "What the agent noticed on its own" },
  { key: "compare", label: "Compare", hint: "Is the difference real?" },
  { key: "trend", label: "Trend", hint: "How it moves over time" },
];

const KIND_TONE: Record<string, "indigo" | "green" | "amber" | "red" | "cyan" | "slate"> = {
  correlation: "indigo",
  segment_difference: "green",
  trend: "cyan",
  anomaly: "amber",
  imbalance: "amber",
  data_quality: "red",
  outliers: "amber",
  skew: "slate",
};

function fmt(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    if (Number.isInteger(value)) return value.toLocaleString();
    return Math.abs(value) >= 1000
      ? value.toLocaleString(undefined, { maximumFractionDigits: 0 })
      : value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  }
  return String(value);
}

/* ------------------------------------------------------------------ table */
function ResultTable({ table }: { table: QueryTable }) {
  if (!table.rows.length) {
    return <p className="text-sm text-slate-500 dark:text-slate-400">{table.note || "No rows."}</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left dark:border-slate-800">
            {table.columns.map((c) => (
              <th key={c} className="px-3 py-2 font-medium text-slate-500 dark:text-slate-400">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i} className="border-b border-slate-100 last:border-0 dark:border-slate-800/60">
              {table.columns.map((c) => (
                <td key={c} className="px-3 py-2 text-slate-800 dark:text-slate-200">
                  {fmt(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {table.truncated && (
        <p className="px-3 py-2 text-xs text-slate-500 dark:text-slate-400">
          Showing {table.rows.length} of {table.row_count.toLocaleString()} rows.
        </p>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------- ask */
function AskTab({ datasetId }: { datasetId: string }) {
  const [question, setQuestion] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [result, setResult] = useState<AskResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .suggestedQuestions(datasetId)
      .then((r) => setSuggestions(r.questions))
      .catch(() => setSuggestions([]));
  }, [datasetId]);

  const run = useCallback(
    async (text: string) => {
      const q = text.trim();
      if (!q) return;
      setBusy(true);
      setError(null);
      try {
        setResult(await api.ask(datasetId, q));
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Could not answer that question.");
      } finally {
        setBusy(false);
      }
    },
    [datasetId],
  );

  return (
    <div className="space-y-4">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run(question);
        }}
        className="flex flex-col gap-2 sm:flex-row"
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. Which region has the highest average revenue?"
          className="flex-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none placeholder:text-slate-400 focus:border-indigo-400 dark:border-slate-700 dark:bg-slate-950 dark:text-white"
        />
        <Button type="submit" disabled={busy || !question.trim()}>
          {busy ? "Working…" : "Ask"}
        </Button>
      </form>

      {suggestions.length > 0 && !result && (
        <div className="flex flex-wrap gap-2">
          {suggestions.map((s) => (
            <button
              key={s}
              onClick={() => {
                setQuestion(s);
                run(s);
              }}
              className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 transition hover:border-indigo-400 hover:text-indigo-600 dark:border-slate-700 dark:text-slate-300 dark:hover:text-indigo-300"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}
      {busy && <Spinner label="Computing with pandas…" />}

      {result && !busy && (
        <div className="space-y-3">
          <Card className="p-4">
            <p
              className="text-sm text-slate-800 dark:text-slate-200"
              dangerouslySetInnerHTML={{
                __html: result.answer.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>"),
              }}
            />
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Badge tone={result.route === "llm" ? "indigo" : "slate"}>
                {result.route === "llm" ? "LLM-planned query" : "Rule-based query"}
              </Badge>
              {(result.table.spec?.group_by ?? []).map((g: string) => (
                <Badge key={g} tone="cyan">
                  group by {g}
                </Badge>
              ))}
              {(result.table.spec?.filters ?? []).map((f: any, i: number) => (
                <Badge key={i} tone="amber">
                  {f.column} {f.op} {String(f.value)}
                </Badge>
              ))}
            </div>
          </Card>

          {result.chart && (
            <Card className="p-4">
              <PlotlyChart figure={result.chart} />
            </Card>
          )}

          <Card className="p-1">
            <ResultTable table={result.table} />
          </Card>
        </div>
      )}
    </div>
  );
}

/* --------------------------------------------------------------- insights */
function InsightCard({ insight }: { insight: Insight }) {
  return (
    <Card className="p-4">
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-white">{insight.title}</h3>
        <Badge tone={KIND_TONE[insight.kind] ?? "slate"}>{insight.kind.replace(/_/g, " ")}</Badge>
      </div>
      <p className="mt-1.5 text-sm text-slate-600 dark:text-slate-400">{insight.detail}</p>
      <div className="mt-3 flex items-center gap-2">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
          <div
            className="h-full rounded-full bg-indigo-500"
            style={{ width: `${insight.importance}%` }}
          />
        </div>
        <span className="text-xs tabular-nums text-slate-500 dark:text-slate-400">
          {insight.importance.toFixed(0)}
        </span>
      </div>
    </Card>
  );
}

function InsightsTab({ datasetId }: { datasetId: string }) {
  const [data, setData] = useState<InsightsResult | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setBusy(true);
    api
      .insights(datasetId)
      .then(setData)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Could not load insights."))
      .finally(() => setBusy(false));
  }, [datasetId]);

  if (busy) return <Spinner label="Scanning the dataset…" />;
  if (error) return <ErrorAlert message={error} />;
  if (!data || !data.insights.length)
    return <EmptyState title="Nothing unusual stood out" subtitle="No strong patterns, imbalances or quality issues were found." />;

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600 dark:text-slate-400">
        {data.count} finding{data.count === 1 ? "" : "s"}, ranked by how much they matter.
      </p>
      <div className="grid gap-3 md:grid-cols-2">
        {data.insights.map((i, idx) => (
          <InsightCard key={`${i.kind}-${idx}`} insight={i} />
        ))}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- compare */
function CompareTab({ datasetId, schema }: { datasetId: string; schema: DatasetSchema | null }) {
  const [dimension, setDimension] = useState("");
  const [metric, setMetric] = useState("");
  const [result, setResult] = useState<SegmentResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (schema && !dimension) {
      setDimension(schema.dimensions[0] ?? "");
      setMetric(schema.measures[0] ?? "");
    }
  }, [schema, dimension]);

  const run = async () => {
    if (!dimension) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.segments(datasetId, dimension, metric || null));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not compare those columns.");
    } finally {
      setBusy(false);
    }
  };

  if (!schema) return <Spinner />;
  if (!schema.dimensions.length)
    return <EmptyState title="No groupable columns" subtitle="This dataset has no low-cardinality column to compare across." />;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-2">
        <label className="text-xs text-slate-500 dark:text-slate-400">
          Group by
          <select
            value={dimension}
            onChange={(e) => setDimension(e.target.value)}
            className="mt-1 block rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-white"
          >
            {schema.dimensions.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500 dark:text-slate-400">
          Measure
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            className="mt-1 block rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-white"
          >
            <option value="">(row counts)</option>
            {schema.measures.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>
        <Button onClick={run} disabled={busy}>{busy ? "Testing…" : "Compare"}</Button>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {result && !busy && (
        <div className="space-y-3">
          <Card className="p-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={result.test.significant ? "green" : "slate"}>
                {result.test.significant ? "Statistically significant" : "Within noise"}
              </Badge>
              <Badge tone="indigo">{result.test.name}</Badge>
              {result.test.robust_check && (
                <Badge tone={result.test.robust_check.agrees ? "green" : "amber"}>
                  {result.test.robust_check.name} {result.test.robust_check.agrees ? "agrees" : "disagrees"}
                </Badge>
              )}
            </div>
            <p className="mt-2 text-sm text-slate-800 dark:text-slate-200">{result.verdict}</p>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              A p-value at or below {result.test.alpha} means a gap this large is unlikely to be chance alone.
            </p>
          </Card>

          <Card className="p-1">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-left dark:border-slate-800">
                    {Object.keys(result.segments[0] ?? {}).map((k) => (
                      <th key={k} className="px-3 py-2 font-medium text-slate-500 dark:text-slate-400">
                        {k}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.segments.map((s, i) => (
                    <tr key={i} className="border-b border-slate-100 last:border-0 dark:border-slate-800/60">
                      {Object.keys(result.segments[0] ?? {}).map((k) => (
                        <td key={k} className="px-3 py-2 text-slate-800 dark:text-slate-200">
                          {Array.isArray(s[k]) ? s[k].join(" / ") : fmt(s[k])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ trend */
function TrendTab({ datasetId, schema }: { datasetId: string; schema: DatasetSchema | null }) {
  const [dateColumn, setDateColumn] = useState("");
  const [valueColumn, setValueColumn] = useState("");
  const [agg, setAgg] = useState("mean");
  const [result, setResult] = useState<TrendResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (schema && !dateColumn) {
      setDateColumn(schema.date_columns[0] ?? "");
      setValueColumn(schema.measures[0] ?? "");
    }
  }, [schema, dateColumn]);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api.trend(datasetId, {
          date_column: dateColumn || null,
          value_column: valueColumn || null,
          agg,
        }),
      );
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not build a timeline.");
    } finally {
      setBusy(false);
    }
  };

  if (!schema) return <Spinner />;
  if (!schema.date_columns.length)
    return (
      <EmptyState
        title="No date column found"
        subtitle="Trend analysis needs a column of dates. Upload data with a date field to use this tab."
      />
    );

  const chart = result
    ? {
        id: "trend",
        kind: "line",
        title: `${result.value_column ?? "rows"} over time`,
        data: {
          data: [
            {
              type: "scatter",
              mode: "lines",
              name: result.value_column ?? "value",
              x: result.points.map((p) => p.period),
              y: result.points.map((p) => p.value),
              line: { color: "#6366f1" },
            },
            {
              type: "scatter",
              mode: "lines",
              name: "moving average",
              x: result.points.map((p) => p.period),
              y: result.moving_average,
              line: { color: "#f59e0b", dash: "dot" },
            },
          ],
          layout: { margin: { l: 46, r: 18, t: 14, b: 42 }, height: 340 },
        },
      }
    : null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-2">
        <label className="text-xs text-slate-500 dark:text-slate-400">
          Date column
          <select
            value={dateColumn}
            onChange={(e) => setDateColumn(e.target.value)}
            className="mt-1 block rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-white"
          >
            {schema.date_columns.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500 dark:text-slate-400">
          Measure
          <select
            value={valueColumn}
            onChange={(e) => setValueColumn(e.target.value)}
            className="mt-1 block rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-white"
          >
            <option value="">(row counts)</option>
            {schema.measures.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500 dark:text-slate-400">
          Aggregate
          <select
            value={agg}
            onChange={(e) => setAgg(e.target.value)}
            className="mt-1 block rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 dark:border-slate-700 dark:bg-slate-950 dark:text-white"
          >
            {["sum", "mean", "median", "count", "min", "max"].map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </label>
        <Button onClick={run} disabled={busy}>{busy ? "Building…" : "Analyse"}</Button>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {result && !busy && (
        <div className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Card className="p-4">
              <div className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">Direction</div>
              <div className="mt-1 text-xl font-semibold capitalize text-slate-900 dark:text-white">
                {result.trend.direction}
              </div>
              <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                R² {result.trend.r_squared.toFixed(2)}
                {result.trend.significant ? " · significant" : " · not significant"}
              </div>
            </Card>
            <Card className="p-4">
              <div className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">Total change</div>
              <div className="mt-1 text-xl font-semibold text-slate-900 dark:text-white">
                {result.total_change_pct === null ? "—" : `${result.total_change_pct > 0 ? "+" : ""}${result.total_change_pct}%`}
              </div>
              <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                {result.first.period} → {result.last.period}
              </div>
            </Card>
            <Card className="p-4">
              <div className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">Peak</div>
              <div className="mt-1 text-xl font-semibold text-slate-900 dark:text-white">{fmt(result.peak.value)}</div>
              <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{result.peak.period}</div>
            </Card>
            <Card className="p-4">
              <div className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">Anomalies</div>
              <div className="mt-1 text-xl font-semibold text-slate-900 dark:text-white">{result.anomalies.length}</div>
              <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                {result.anomalies.length ? result.anomalies.map((a) => a.period).slice(0, 2).join(", ") : "none flagged"}
              </div>
            </Card>
          </div>

          {chart && (
            <Card className="p-4">
              <PlotlyChart figure={chart as any} />
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ panel */
export default function AnalystPanel({ datasetId }: { datasetId: string }) {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [schema, setSchema] = useState<DatasetSchema | null>(null);

  useEffect(() => {
    api.schema(datasetId).then(setSchema).catch(() => setSchema(null));
  }, [datasetId]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-1 border-b border-slate-200 dark:border-slate-800">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            title={t.hint}
            className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium transition ${
              tab === t.key
                ? "border-indigo-500 text-indigo-600 dark:text-indigo-400"
                : "border-transparent text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "dashboard" && <DashboardPanel datasetId={datasetId} />}
      {tab === "ask" && <AskTab datasetId={datasetId} />}
      {tab === "insights" && <InsightsTab datasetId={datasetId} />}
      {tab === "compare" && <CompareTab datasetId={datasetId} schema={schema} />}
      {tab === "trend" && <TrendTab datasetId={datasetId} schema={schema} />}
    </div>
  );
}
