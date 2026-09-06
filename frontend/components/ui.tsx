"use client";

import type { ReactNode } from "react";

/* ------------------------------------------------------------------ */
export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-800 dark:bg-slate-900 ${className}`}
    >
      {children}
    </div>
  );
}

export function Section({
  title,
  subtitle,
  action,
  children,
  id,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  id?: string;
}) {
  return (
    <section id={id} className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            {title}
          </h2>
          {subtitle && <p className="mt-0.5 text-sm text-slate-500 dark:text-slate-400">{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

export function StatCard({
  label,
  value,
  hint,
  accent = false,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  accent?: boolean;
}) {
  return (
    <Card className="p-4">
      <div className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
        {label}
      </div>
      <div
        className={`mt-1 truncate text-2xl font-semibold ${
          accent ? "text-indigo-600 dark:text-indigo-400" : "text-slate-900 dark:text-white"
        }`}
        title={typeof value === "string" ? value : undefined}
      >
        {value}
      </div>
      {hint && <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{hint}</div>}
    </Card>
  );
}

export function Badge({
  children,
  tone = "slate",
}: {
  children: ReactNode;
  tone?: "slate" | "indigo" | "green" | "amber" | "red" | "cyan";
}) {
  const tones: Record<string, string> = {
    slate: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    indigo: "bg-indigo-50 text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-300",
    green: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300",
    amber: "bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300",
    red: "bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300",
    cyan: "bg-cyan-50 text-cyan-700 dark:bg-cyan-500/15 dark:text-cyan-300",
  };
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
      <svg className="h-4 w-4 animate-spin text-indigo-500" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
      </svg>
      {label}
    </span>
  );
}

export function ErrorAlert({ message, onDismiss }: { message: string; onDismiss?: () => void }) {
  return (
    <div className="flex items-start justify-between gap-3 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800 dark:border-rose-500/30 dark:bg-rose-500/10 dark:text-rose-200">
      <div>
        <span className="font-semibold">Something went wrong.</span> {message}
      </div>
      {onDismiss && (
        <button onClick={onDismiss} className="text-rose-500 hover:text-rose-700" aria-label="Dismiss">
          ✕
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-slate-300 py-14 text-center dark:border-slate-700">
      <div className="text-3xl">📊</div>
      <div className="font-medium text-slate-700 dark:text-slate-200">{title}</div>
      {subtitle && <div className="max-w-md text-sm text-slate-500 dark:text-slate-400">{subtitle}</div>}
      {action}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  className = "",
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  disabled?: boolean;
  className?: string;
  type?: "button" | "submit";
}) {
  const styles: Record<string, string> = {
    primary:
      "bg-indigo-600 text-white hover:bg-indigo-500 disabled:bg-indigo-300 dark:disabled:bg-indigo-900 shadow-sm",
    secondary:
      "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800",
    ghost: "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800",
    danger: "bg-rose-600 text-white hover:bg-rose-500",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-60 ${styles[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

export function ProgressBar({
  label,
  sub,
  value,
}: {
  label: string;
  sub?: string;
  /** 0–1 when the real position is known; omit for an indeterminate bar. */
  value?: number;
}) {
  const known = typeof value === "number" && Number.isFinite(value);
  const pct = known ? Math.max(2, Math.min(100, value * 100)) : 50;
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="font-medium text-slate-700 dark:text-slate-200">{label}</span>
        {sub && <span className="shrink-0 text-xs text-slate-500">{sub}</span>}
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
        <div
          className={`h-full rounded-full bg-indigo-500 ${known ? "transition-all duration-500" : "animate-pulse"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
