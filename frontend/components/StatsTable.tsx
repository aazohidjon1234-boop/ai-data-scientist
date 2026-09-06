"use client";

import type { Analysis } from "@/lib/types";
import { fmtNum } from "@/lib/format";
import { Card } from "./ui";

export default function StatsTable({ analysis }: { analysis: Analysis }) {
  const stats = analysis.statistics;
  const numeric = Object.entries(stats).filter(([, v]) => v.kind === "numeric") as [
    string,
    Extract<(typeof stats)[string], { kind: "numeric" }>,
  ][];
  const categorical = Object.entries(stats).filter(([, v]) => v.kind === "categorical") as [
    string,
    Extract<(typeof stats)[string], { kind: "categorical" }>,
  ][];

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card>
        <div className="border-b border-slate-100 px-4 py-2.5 text-sm font-semibold dark:border-slate-800">
          Numeric columns
        </div>
        <div className="thin-scroll max-h-[360px] overflow-auto">
          <table className="min-w-full text-sm">
            <thead className="sticky top-0 bg-slate-50 text-left dark:bg-slate-800">
              <tr>
                {["Column", "Mean", "Median", "Std", "Min", "Q1", "Q3", "Max"].map((h) => (
                  <th key={h} className="whitespace-nowrap px-3 py-2 font-semibold text-slate-600 dark:text-slate-300">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {numeric.map(([name, v]) => (
                <tr key={name} className="border-t border-slate-100 dark:border-slate-800">
                  <td className="whitespace-nowrap px-3 py-1.5 font-medium text-slate-800 dark:text-slate-100">{name}</td>
                  {[v.mean, v.median, v.std, v.min, v.q1, v.q3, v.max].map((n, i) => (
                    <td key={i} className="whitespace-nowrap px-3 py-1.5 tabular-nums text-slate-600 dark:text-slate-300">
                      {fmtNum(n)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <div className="border-b border-slate-100 px-4 py-2.5 text-sm font-semibold dark:border-slate-800">
          Categorical columns
        </div>
        <div className="divide-y divide-slate-100 dark:divide-slate-800">
          {categorical.length === 0 && (
            <div className="px-4 py-6 text-sm text-slate-500">No categorical columns.</div>
          )}
          {categorical.map(([name, v]) => (
            <div key={name} className="px-4 py-2.5">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-slate-800 dark:text-slate-100">{name}</span>
                <span className="text-xs text-slate-400">{v.unique} unique values</span>
              </div>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {v.top_values.map((t) => (
                  <span
                    key={t.value}
                    className="rounded-md bg-slate-100 px-2 py-0.5 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                  >
                    {t.value} · {t.count}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
