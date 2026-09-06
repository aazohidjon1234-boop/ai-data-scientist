export function fmtNum(v: unknown): string {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  if (!isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (abs >= 10_000) return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (abs >= 100) return n.toLocaleString("en-US", { maximumFractionDigits: 1 });
  if (abs >= 1) return n.toLocaleString("en-US", { maximumFractionDigits: 3 });
  return n.toLocaleString("en-US", { maximumFractionDigits: 4 });
}

export function timeAgo(iso: string): string {
  const then = new Date(iso).getTime();
  const s = Math.max(1, Math.round((Date.now() - then) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}

export function taskLabel(t: string | null | undefined): string {
  switch (t) {
    case "regression":
      return "Regression";
    case "classification":
      return "Classification";
    case "clustering":
      return "Clustering";
    default:
      return "—";
  }
}

export function metricLabel(key: string): string {
  const map: Record<string, string> = {
    mae: "MAE",
    mse: "MSE",
    rmse: "RMSE",
    r2: "R²",
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
    f1: "F1",
    silhouette: "Silhouette",
    inertia: "Inertia",
    k: "k",
  };
  return map[key] || key.toUpperCase();
}
