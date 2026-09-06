"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Analysis, Dataset, ModelRun, ProblemType, ReportInfo } from "@/lib/types";
import { taskLabel, timeAgo } from "@/lib/format";
import { Badge, Button, Card, ErrorAlert, ProgressBar, Section, StatCard, Spinner } from "@/components/ui";
import DataTable from "@/components/DataTable";
import StatsTable from "@/components/StatsTable";
import ModelsSection from "@/components/ModelsSection";
import AgentTimeline from "@/components/AgentTimeline";
import ChatPanel from "@/components/ChatPanel";
import AnalystPanel from "@/components/AnalystPanel";
import ImprovePanel from "@/components/ImprovePanel";
import TargetSelector from "@/components/TargetSelector";
import PlotlyChart from "@/components/PlotlyChart";
import Markdown from "@/components/Markdown";

export default function AnalysisPage({ params }: { params: { id: string } }) {
  const id = params.id;

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [run, setRun] = useState<ModelRun | null>(null);
  const [busy, setBusy] = useState<null | "analyze" | "train">(null);
  const [error, setError] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [report, setReport] = useState<ReportInfo | null>(null);
  const [tab, setTab] = useState<"models" | "activity" | "report">("models");
  const didAutoRun = useRef(false);

  useEffect(() => {
    if (!startedAt) return;
    const t = setInterval(() => setElapsed(Math.round((Date.now() - startedAt) / 1000)), 500);
    return () => clearInterval(t);
  }, [startedAt]);

  const doAnalyze = useCallback(
    async (target?: string | null) => {
      if (!id) return null;
      setBusy("analyze");
      setStartedAt(Date.now());
      setElapsed(0);
      setError(null);
      try {
        const a = await api.analyze(id, target);
        setAnalysis(a);
        const d = await api.getDataset(id);
        setDataset(d.dataset);
        return a;
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Analysis failed.");
        return null;
      } finally {
        setBusy(null);
        setStartedAt(null);
      }
    },
    [id]
  );

  const doTrain = useCallback(
    async (payload: {
      target?: string | null;
      problem_type?: ProblemType | null;
      features?: string[] | null;
      drop_outliers?: boolean;
    }) => {
      if (!id) return null;
      setBusy("train");
      setStartedAt(Date.now());
      setElapsed(0);
      setError(null);
      try {
        const t = await api.train(id, payload);
        setRun(t);
        const d = await api.getDataset(id);
        setDataset(d.dataset);
        return t;
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Training failed.");
        return null;
      } finally {
        setBusy(null);
        setStartedAt(null);
      }
    },
    [id]
  );

  // load + auto-run the pipeline
  useEffect(() => {
    if (!id || didAutoRun.current) return;
    didAutoRun.current = true;
    (async () => {
      try {
        const detail = await api.getDataset(id);
        setDataset(detail.dataset);
        let a = detail.analysis;
        if (!a) {
          a = await doAnalyze();
        }
        setAnalysis(a);
        if (a && !detail.models) {
          if (a.target_column) {
            await doTrain({});
          }
          // no target -> show the target selector instead of auto-training
        } else if (detail.models) {
          setRun(detail.models);
        }
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Failed to load this analysis.");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  if (!id) return null;

  if (!dataset && !error && !busy) {
    return (
      <Card className="p-10">
        <Spinner label="Loading dataset…" />
      </Card>
    );
  }

  const task = analysis?.problem_type ?? dataset?.problem_type ?? null;

  return (
    <div className="space-y-8">
      {/* header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link href="/" className="text-xs font-medium text-indigo-600 hover:underline dark:text-indigo-400">
            ← Back to dashboard
          </Link>
          <h1 className="mt-1 text-2xl font-bold tracking-tight">{dataset?.name ?? "…"}</h1>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            {dataset && (
              <Badge tone="slate">
                {dataset.rows.toLocaleString()} rows × {dataset.columns} cols
              </Badge>
            )}
            {task && <Badge tone="indigo">{taskLabel(task)}</Badge>}
            {analysis?.target_column && (
              <Badge tone="green">target: {analysis.target_column}</Badge>
            )}
            {dataset?.best_model && <Badge tone="cyan">best: {dataset.best_model}</Badge>}
            {dataset && <span className="text-xs text-slate-400">{timeAgo(dataset.created_at)}</span>}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" disabled={!!busy} onClick={() => doAnalyze()}>
            {busy === "analyze" ? "Analyzing…" : "Re-analyze"}
          </Button>
          <Button
            variant="secondary"
            disabled={!!busy || !analysis}
            onClick={() => doTrain({ target: analysis?.target_column ?? null })}
          >
            {busy === "train" ? "Training…" : "Re-train"}
          </Button>
        </div>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      {busy && (
        <Card className="p-5">
          <ProgressBar
            label={
              busy === "analyze"
                ? "Agent is inspecting the dataset (profiling, quality checks, target detection)…"
                : "Agent is cleaning the data and training models…"
            }
            sub={`${elapsed}s elapsed — real computation is running on the server`}
          />
          {busy === "train" && (
            <p className="mt-3 text-xs text-slate-500">
              Cleaning → feature preparation → train/test split → training each model → evaluating →
              comparing.
            </p>
          )}
        </Card>
      )}

      {/* overview */}
      {analysis && dataset && !busy && (
        <Section title="Overview" subtitle="Everything the agent found in the first pass">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <StatCard label="Rows" value={analysis.profile.rows.toLocaleString()} />
            <StatCard label="Columns" value={analysis.profile.columns} />
            <StatCard
              label="Missing values"
              value={analysis.missing.total_missing.toLocaleString()}
              hint={`${analysis.missing.pct_missing}% of cells`}
            />
            <StatCard label="Duplicate rows" value={analysis.profile.duplicate_rows} />
            <StatCard label="Target" value={analysis.target_column || "—"} hint={analysis.target_column ? "auto-detected" : "unsupervised"} />
            <StatCard label="Task" value={taskLabel(task)} accent />
          </div>
        </Section>
      )}

      {/* Training configuration. Shown after a run too, so the target and the
          input columns can be changed and the run repeated. */}
      {run && analysis && !busy && (
        <ImprovePanel
          datasetId={id}
          target={run.target}
          problemType={run.problem_type}
          busy={!!busy}
          onRetrain={(p) => doTrain(p)}
        />
      )}

      {analysis && !busy && (
        <TargetSelector
          analysis={analysis}
          busy={!!busy}
          hasRun={!!run}
          datasetId={id}
          onTrain={(p) => doTrain(p)}
        />
      )}

      {/* data preview */}
      {analysis && !busy && (
        <Section title="Data preview" subtitle="First rows of the original upload">
          <DataTable analysis={analysis} />
        </Section>
      )}

      {/* statistics */}
      {analysis && !busy && (
        <Section title="Statistics" subtitle="Computed with pandas">
          <StatsTable analysis={analysis} />
        </Section>
      )}

      {/* analyst workspace — ad-hoc questions, insights, comparisons, trends */}
      {analysis && !busy && (
        <Section
          title="Analyst"
          subtitle="Question the data directly — every answer is computed with pandas, not written by the model"
        >
          <AnalystPanel datasetId={id} />
        </Section>
      )}

      {/* visualizations */}
      {analysis && !busy && analysis.figures.length > 0 && (
        <Section title="Visualizations" subtitle="Generated with Plotly from the actual data">
          <div className="grid gap-4 lg:grid-cols-2">
            {analysis.figures.map((f) => (
              <Card key={f.id} className="p-4">
                <h3 className="mb-1 text-sm font-semibold text-slate-700 dark:text-slate-200">{f.title}</h3>
                <PlotlyChart figure={f} />
              </Card>
            ))}
          </div>
        </Section>
      )}

      {/* models + tabs */}
      {(run || busy === "train") && (
        <Section title="Model training" subtitle={run ? `Target: ${run.target ?? "—"} · ${taskLabel(run.problem_type)}` : "In progress…"}>
          {run && !busy && (
            <>
              <div className="mb-4 flex flex-wrap gap-2">
                {(["models", "activity", "report"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`rounded-lg px-3.5 py-1.5 text-sm font-medium capitalize transition ${
                      tab === t
                        ? "bg-indigo-600 text-white"
                        : "bg-white text-slate-600 hover:bg-slate-100 dark:bg-slate-900 dark:text-slate-300 dark:hover:bg-slate-800"
                    }`}
                  >
                    {t === "activity" ? "Agent activity" : t}
                  </button>
                ))}
              </div>

              {tab === "models" && <ModelsSection run={run} />}

              {tab === "activity" && (
                <div className="grid gap-6 lg:grid-cols-2">
                  <Card className="p-5">
                    <AgentTimeline steps={analysis?.trace ?? []} title="Analysis tools" />
                  </Card>
                  <Card className="p-5">
                    <AgentTimeline steps={run.trace} title="Training tools" />
                  </Card>
                </div>
              )}

              {tab === "report" && (
                <ReportPanel datasetId={id} hasReport={!!report} onGenerated={setReport} report={report} />
              )}
            </>
          )}
        </Section>
      )}

      {/* AI insights */}
      {analysis && !busy && (
        <Section
          title="AI insights"
          subtitle="The agent explains the analysis and answers your questions — grounded only in the computed results"
        >
          <Card className="p-5">
            <ChatPanel
              datasetId={id}
              initialMessage={run?.explanation || analysis.explanation}
            />
          </Card>
        </Section>
      )}

      {/* agent plan (pre-training) */}
      {analysis && !run && !busy && analysis.plan.length > 0 && (
        <Section title="Agent's plan">
          <Card className="p-5">
            <ol className="list-decimal space-y-1.5 pl-5 text-sm text-slate-600 dark:text-slate-300">
              {analysis.plan.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ol>
          </Card>
        </Section>
      )}

      {/* footer note */}
      <p className="text-xs text-slate-400">
        All metrics on this page are computed by the Python pipeline (pandas, scikit-learn, Plotly).
        The language model only interprets the numbers — it cannot invent them.
      </p>
    </div>
  );
}

function ReportPanel({
  datasetId,
  report,
  onGenerated,
}: {
  datasetId: string;
  report: ReportInfo | null;
  onGenerated: (r: ReportInfo) => void;
  busy?: boolean;
  hasReport?: boolean;
}) {
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function generate() {
    setErr(null);
    setBusy(true);
    try {
      const r = await api.report(datasetId);
      onGenerated(r);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : "Report generation failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-slate-500 dark:text-slate-400">
          Full Markdown report with overview, data quality, statistics, model comparison and the AI
          interpretation.
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={generate} disabled={busy}>
            {busy ? "Generating…" : report ? "Regenerate report" : "Generate report"}
          </Button>
          {report && (
            <a
              href={report.download_url}
              download
              className="inline-flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500"
            >
              ⬇ Download .md
            </a>
          )}
        </div>
      </div>
      {err && <div className="mt-3 text-sm text-rose-500">{err}</div>}
      {report && (
        <div className="thin-scroll mt-4 max-h-[480px] overflow-auto rounded-lg border border-slate-200 bg-slate-50 p-5 dark:border-slate-800 dark:bg-slate-950/60">
          <Markdown text={report.content} />
        </div>
      )}
    </Card>
  );
}
