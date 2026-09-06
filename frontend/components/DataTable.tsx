"use client";

import type { Analysis } from "@/lib/types";
import { fmtNum } from "@/lib/format";
import { Card } from "./ui";

export default function DataTable({ analysis }: { analysis: Analysis }) {
  const { columns, dtypes, rows } = analysis.preview;
  return (
    <Card>
      <div className="thin-scroll max-h-[420px] overflow-auto">
        <table className="min-w-full text-sm">
          <thead className="sticky top-0 z-10">
            <tr className="bg-slate-50 text-left dark:bg-slate-800">
              {columns.map((c, i) => (
                <th key={c} className="whitespace-nowrap border-b border-slate-200 px-3 py-2 dark:border-slate-700">
                  <span className="font-semibold text-slate-700 dark:text-slate-200">{c}</span>
                  <span className="ml-1.5 text-[10px] font-normal text-slate-400">{dtypes[i]}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => (
              <tr key={ri} className="border-b border-slate-100 last:border-0 hover:bg-slate-50/60 dark:border-slate-800 dark:hover:bg-slate-800/40">
                {row.map((cell, ci) => (
                  <td key={ci} className="whitespace-nowrap px-3 py-1.5 text-slate-600 dark:text-slate-300">
                    {cell === null ? (
                      <span className="italic text-slate-400">NaN</span>
                    ) : typeof cell === "number" ? (
                      fmtNum(cell)
                    ) : (
                      cell
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="border-t border-slate-100 px-4 py-2 text-xs text-slate-400 dark:border-slate-800">
        Showing first {rows.length} of {analysis.profile.rows.toLocaleString()} rows
      </div>
    </Card>
  );
}
