"use client";

import dynamic from "next/dynamic";
import { useTheme } from "@/lib/theme";
import type { Figure } from "@/lib/types";

const Plot = dynamic(
  () => import("react-plotly.js").then((m) => m.default),
  {
    ssr: false,
    loading: () => (
      <div className="h-[320px] w-full animate-pulse rounded-lg bg-slate-100 dark:bg-slate-800" />
    ),
  }
);

export default function PlotlyChart({ figure, height = 320 }: { figure: Figure; height?: number }) {
  const { theme } = useTheme();
  const dark = theme === "dark";

  const layout: Record<string, unknown> = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: dark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0)",
    font: { color: dark ? "#cbd5e1" : "#334155", size: 12 },
    colorway: dark ? ["#818cf8", "#34d399", "#fbbf24", "#f87171", "#22d3ee", "#c084fc"] : ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#a855f7"],
    ...(figure.data.layout || {}),
    height,
  };

  return (
    <Plot
      data={figure.data.data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height }}
      useResizeHandler
    />
  );
}
