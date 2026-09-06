"use client";

import { useState } from "react";
import type { TraceStep } from "@/lib/types";
import { TOOL_DESCRIPTIONS } from "@/lib/toolDescriptions";

function StatusIcon({ status }: { status: TraceStep["status"] }) {
  if (status === "ok")
    return <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-100 text-[11px] text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300">✓</span>;
  if (status === "failed")
    return <span className="flex h-5 w-5 items-center justify-center rounded-full bg-rose-100 text-[11px] text-rose-700 dark:bg-rose-500/20 dark:text-rose-300">✕</span>;
  return <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-100 text-[11px] text-slate-500 dark:bg-slate-800">–</span>;
}

export default function AgentTimeline({ steps, title }: { steps: TraceStep[]; title: string }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-slate-700 dark:text-slate-200">{title}</h3>
      <ol className="relative space-y-3 border-l border-slate-200 pl-6 dark:border-slate-800">
        {steps.map((s, i) => {
          const key = `${s.tool}-${i}`;
          const desc = TOOL_DESCRIPTIONS[s.tool] || s.tool;
          return (
            <li key={key} className="relative">
              <span className="absolute -left-[31px] top-0.5">
                <StatusIcon status={s.status} />
              </span>
              <button
                onClick={() => setOpen(open === key ? null : key)}
                className="w-full text-left"
              >
                <div className="flex flex-wrap items-baseline gap-x-2">
                  <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs font-semibold text-indigo-700 dark:bg-slate-800 dark:text-indigo-300">
                    {s.tool}()
                  </code>
                  <span className="text-xs text-slate-400">{desc}</span>
                  <span className="ml-auto text-[11px] tabular-nums text-slate-400">
                    {s.duration_s > 0 ? `${s.duration_s.toFixed(2)}s` : ""}
                  </span>
                </div>
                {s.observation && (
                  <p className="mt-1 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                    {s.observation}
                  </p>
                )}
                {s.status === "failed" && s.error && (
                  <p className="mt-1 text-xs text-rose-600 dark:text-rose-400">{s.error}</p>
                )}
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
