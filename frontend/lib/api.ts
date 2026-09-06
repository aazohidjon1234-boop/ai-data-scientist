import type {
  Analysis,
  AskResult,
  Dataset,
  DatasetDetail,
  DashboardResult,
  DatasetSchema,
  FeatureSuggestion,
  Figure,
  ImproveResult,
  InsightsResult,
  ModelRun,
  PipelineProgress,
  ProblemType,
  ReportInfo,
  SampleInfo,
  SegmentResult,
  TrendResult,
} from "./types";

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(message: string, status: number, code: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError("Cannot reach the API. Is the backend running?", 0, "network");
  }
  const body = (await res.json().catch(() => ({}))) as Record<string, any>;
  if (!res.ok) {
    const err = body?.error || {};
    throw new ApiError(err.message || `Request failed (${res.status})`, res.status, err.code || "error");
  }
  return body as T;
}

export const api = {
  health: () => http<{ status: string; llm_enabled: boolean }>("/api/health"),
  listDatasets: () => http<Dataset[]>("/api/datasets"),
  getDataset: (id: string) => http<DatasetDetail>(`/api/datasets/${id}`),
  uploadDataset: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return http<Dataset>("/api/datasets/upload", { method: "POST", body: fd });
  },
  listSamples: () => http<SampleInfo[]>("/api/datasets/samples"),
  loadSample: (name: string) =>
    http<Dataset>("/api/datasets/load-sample", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  analyze: (id: string, target?: string | null) =>
    http<Analysis>(`/api/datasets/${id}/analyze`, {
      method: "POST",
      body: JSON.stringify(target ? { target } : {}),
    }),
  train: (
    id: string,
    payload: {
      target?: string | null;
      problem_type?: ProblemType | null;
      k?: number | null;
      features?: string[] | null;
      drop_outliers?: boolean;
      tune?: boolean;
    },
  ) =>
    http<ModelRun>(`/api/datasets/${id}/train`, { method: "POST", body: JSON.stringify(payload) }),
  progress: (id: string) => http<PipelineProgress>(`/api/datasets/${id}/progress`),
  models: (id: string) => http<ModelRun>(`/api/datasets/${id}/models`),
  visualizations: (id: string) => http<{ dataset_id: string; figures: Figure[] }>(`/api/datasets/${id}/visualizations`),
  chat: (id: string, message: string) =>
    http<{ reply: string; tool_calls: string[]; source: string }>(`/api/datasets/${id}/chat`, {
      method: "POST",
      body: JSON.stringify({ message }),
    }),
  report: (id: string) =>
    http<ReportInfo>(`/api/datasets/${id}/report`, { method: "POST" }),

  /* ------------------------------------------------------------ analyst */
  schema: (id: string) => http<DatasetSchema>(`/api/datasets/${id}/schema`),
  suggestedQuestions: (id: string) =>
    http<{ questions: string[] }>(`/api/datasets/${id}/suggested-questions`),
  ask: (id: string, question: string) =>
    http<AskResult>(`/api/datasets/${id}/ask`, {
      method: "POST",
      body: JSON.stringify({ question }),
    }),
  dashboard: (
    id: string,
    payload: {
      filters?: { column: string; op: string; value: unknown }[];
      measure?: string | null;
      dimension?: string | null;
      date_column?: string | null;
    } = {},
  ) =>
    http<DashboardResult>(`/api/datasets/${id}/dashboard`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  improve: (id: string) =>
    http<ImproveResult>(`/api/datasets/${id}/improve`, { method: "POST", body: "{}" }),
  suggestFeatures: (id: string, target?: string | null, problem_type?: ProblemType | null) =>
    http<FeatureSuggestion>(`/api/datasets/${id}/suggest-features`, {
      method: "POST",
      body: JSON.stringify({ target: target || null, problem_type: problem_type || null }),
    }),
  insights: (id: string, limit = 12) =>
    http<InsightsResult>(`/api/datasets/${id}/insights?limit=${limit}`),
  segments: (id: string, dimension: string, metric?: string | null) =>
    http<SegmentResult>(`/api/datasets/${id}/segments`, {
      method: "POST",
      body: JSON.stringify({ dimension, metric: metric || null }),
    }),
  trend: (
    id: string,
    payload: { date_column?: string | null; value_column?: string | null; agg?: string },
  ) =>
    http<TrendResult>(`/api/datasets/${id}/trend`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
