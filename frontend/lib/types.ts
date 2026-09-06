export type ProblemType = "regression" | "classification" | "clustering";

export interface Dataset {
  id: string;
  name: string;
  original_filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  source: string;
  status: string;
  created_at: string;
  has_analysis: boolean;
  has_training: boolean;
  best_model: string | null;
  problem_type: ProblemType | null;
  target: string | null;
}

export interface Figure {
  id: string;
  kind: string;
  title: string;
  data: { data: unknown[]; layout: Record<string, unknown> };
}

export interface TraceStep {
  tool: string;
  args: Record<string, unknown>;
  status: "ok" | "failed" | "skipped";
  observation: string;
  duration_s: number;
  error?: string;
}

export interface Analysis {
  id: string;
  dataset_id: string;
  target_column: string | null;
  problem_type: ProblemType;
  target_candidates: { column: string; kind: string; unique: number; score: number; suggested_task: string }[];
  target_detection: Record<string, unknown>;
  profile: {
    rows: number;
    columns: number;
    numeric_columns: string[];
    categorical_columns: string[];
    duplicate_rows: number;
    all_nan_columns: string[];
    columns_info: { name: string; dtype: string; kind: string; unique: number; missing: number }[];
    memory_mb: number;
  };
  missing: {
    total_missing: number;
    pct_missing: number;
    columns_affected: number;
    per_column: { column: string; missing: number; pct: number; imputation: string }[];
  };
  statistics: Record<
    string,
    | {
        kind: "numeric";
        mean: number;
        median: number;
        std: number;
        min: number;
        q1: number;
        q3: number;
        max: number;
      }
    | { kind: "categorical"; unique: number; top_values: { value: string; count: number }[] }
  >;
  correlation: { columns: string[]; matrix: number[][]; top_pairs: { a: string; b: string; corr: number }[] } | null;
  outliers: { columns: Record<string, { outliers: number; pct: number; bounds: [number, number] }>; total_outliers: number };
  preview: { columns: string[]; dtypes: string[]; rows: (string | number | null)[][] };
  figures: Figure[];
  plan: string[];
  trace: TraceStep[];
  explanation: string;
  created_at: string;
}

export interface DatasetDetail {
  dataset: Dataset;
  analysis: Analysis | null;
  models: ModelRun | null;
}

export interface ModelResult {
  name: string;
  model_type: string;
  status: string;
  status_message: string;
  primary_metric: number | null;
  metrics: Record<string, unknown>;
  feature_importance: { feature: string; importance: number }[] | null;
  training_seconds: number;
  is_best: boolean;
  download_url?: string | null;
  rank?: number | null;
}

export interface ModelRun {
  features_used?: string[] | null;
  source_columns?: string[];
  drop_outliers?: boolean;
  dataset_id: string;
  run_id: string;
  problem_type: ProblemType;
  target: string | null;
  run_info: Record<string, unknown>;
  models: ModelResult[];
  best_model: string | null;
  explanation: string;
  trace: TraceStep[];
  created_at: string;
}

export interface SampleInfo {
  name: string;
  label: string;
  description: string;
  rows: number;
  columns: number;
  task: string;
}

export interface ReportInfo {
  id: string;
  dataset_id: string;
  format: string;
  created_at: string;
  content: string;
  download_url: string;
}

/* ---------------------------------------------------------------- analyst */

export interface QueryTable {
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  rows_after_filter: number;
  truncated: boolean;
  spec: Record<string, any>;
  note?: string;
}

export interface AskResult {
  question: string;
  route: "llm" | "heuristic";
  answer: string;
  table: QueryTable;
  chart: Figure | null;
  tools_used: string[];
}

export interface Insight {
  kind: string;
  title: string;
  detail: string;
  importance: number;
  columns: string[];
  evidence: Record<string, unknown>;
}

export interface InsightsResult {
  dataset_id: string;
  count: number;
  returned: number;
  headline: string;
  by_kind: Record<string, number>;
  insights: Insight[];
  tools_used: string[];
}

export interface SchemaColumn {
  name: string;
  kind: "numeric" | "categorical" | "datetime" | "boolean";
  dtype: string;
  missing: number;
  unique: number;
  min?: number | null;
  max?: number | null;
  examples?: string[];
}

export interface DatasetSchema {
  dataset_id: string;
  rows: number;
  columns: SchemaColumn[];
  dimensions: string[];
  measures: string[];
  date_columns: string[];
}

export interface SegmentTest {
  name: string;
  statistic: number | null;
  p_value: number | null;
  significant: boolean;
  alpha: number;
  dof?: number;
  effect_size?: Record<string, number>;
  robust_check?: { name: string; p_value: number | null; agrees: boolean };
  expected_per_group?: number;
}

export interface SegmentResult {
  dimension: string;
  metric: string | null;
  kind: "numeric" | "categorical" | "distribution";
  segments: Record<string, any>[];
  levels?: string[];
  test: SegmentTest;
  gap?: { best: string; worst: string; difference: number | null };
  verdict: string;
}

export interface TrendPoint {
  period: string;
  value: number | null;
}

export interface TrendResult {
  date_column: string;
  value_column: string | null;
  agg: string;
  freq: string;
  points: TrendPoint[];
  moving_average: (number | null)[];
  period_change_pct: (number | null)[];
  trend: {
    direction: "rising" | "falling" | "flat";
    slope_per_period: number | null;
    r_squared: number;
    p_value: number | null;
    significant: boolean;
  };
  total_change_pct: number | null;
  first: TrendPoint;
  last: TrendPoint;
  peak: TrendPoint;
  trough: TrendPoint;
  anomalies: { period: string; value: number; z_score: number; direction: string }[];
  seasonality: { by_month?: { label: string; value: number }[]; by_weekday?: { label: string; value: number }[] } | null;
}

export interface FeatureVerdict {
  column: string;
  relevance: number;
  verdict: "keep" | "weak" | "review" | "drop";
  reason: string;
  missing_pct: number;
  unique: number;
}

export interface FeatureSuggestion {
  target: string;
  problem_type: ProblemType;
  columns: FeatureVerdict[];
  recommended: string[];
  all_usable: string[];
  evaluation: {
    metric: string;
    folds: number;
    model: string;
    score_all_usable: number | null;
    score_recommended: number | null;
    rows_used: number;
  };
  summary: string;
  tools_used: string[];
}

export interface DashboardKpi {
  label: string;
  value: number | string;
  format: "int" | "num" | "text";
  hint: string;
}

export interface DashboardFilterOption {
  column: string;
  values: string[];
  truncated: boolean;
}

export interface DashboardResult {
  dataset_id: string;
  dataset_name: string;
  layout: {
    measures: string[];
    dimensions: string[];
    date_columns: string[];
    primary_measure: string | null;
    primary_dimension: string | null;
    primary_date: string | null;
  };
  measure: string | null;
  dimension: string | null;
  date_column: string | null;
  rows_total: number;
  rows_shown: number;
  kpis: DashboardKpi[];
  charts: Figure[];
  filter_options: DashboardFilterOption[];
  empty: boolean;
  tools_used: string[];
}

export interface ImproveRecipe {
  key: string;
  label: string;
  why: string;
  score: number | null;
  delta: number | null;
  best_model: string | null;
  rows_used: number;
  features_used: number;
  changes: {
    features: string[] | null;
    drop_outliers: boolean;
    tune?: boolean;
    impute_numeric?: string;
    impute_categorical?: string;
  };
  params?: Record<string, unknown>;
}

export interface ImproveResult {
  imputation?: {
    numeric: string;
    categorical: string;
    columns_with_gaps: string[];
  };
  tuning?: {
    model: string;
    params: Record<string, unknown>;
    cv_score: number;
    metric: string;
    iterations: number;
    folds: number;
  } | null;
  target: string;
  problem_type: ProblemType;
  metric: string;
  folds: number;
  baseline_score: number | null;
  recommended: string;
  recipes: ImproveRecipe[];
  outliers: {
    rows_flagged: number;
    pct: number;
    columns_checked: string[];
    skipped_reason: string | null;
  };
  summary: string;
  tools_used: string[];
}

export interface PipelineProgress {
  running: boolean;
  phase: string | null;
  current: string | null;
  detail: string;
  planned: string[];
  done: { stage: string; observation: string; seconds: number }[];
  completed: number;
  total: number;
  elapsed: number;
  stage_elapsed: number;
  finished: boolean;
  error: string | null;
}
