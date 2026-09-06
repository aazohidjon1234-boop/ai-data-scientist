"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Dataset, SampleInfo } from "@/lib/types";

export default function UploadCard({ onDataset }: { onDataset: (ds: Dataset) => void }) {
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleInfo[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.listSamples().then(setSamples).catch(() => setSamples([]));
  }, []);

  async function handleFile(file: File) {
    setError(null);
    setBusy(true);
    setStage("Uploading & validating…");
    try {
      const ds = await api.uploadDataset(file);
      onDataset(ds);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Upload failed.");
      setStage("");
    } finally {
      setBusy(false);
    }
  }

  async function loadSample(name: string) {
    setError(null);
    setBusy(true);
    setStage(`Loading sample “${name}”…`);
    try {
      const ds = await api.loadSample(name);
      onDataset(ds);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load sample.");
      setStage("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) handleFile(f);
        }}
        className={`flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 text-center transition ${
          dragging
            ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-500/10"
            : "border-slate-300 bg-white hover:border-indigo-300 dark:border-slate-700 dark:bg-slate-900"
        }`}
      >
        <div className="text-4xl">{busy ? "⏳" : "📤"}</div>
        <div>
          <p className="font-medium text-slate-800 dark:text-slate-100">
            {busy ? stage : "Drag & drop a CSV here"}
          </p>
          <p className="mt-0.5 text-sm text-slate-500 dark:text-slate-400">
            or
          </p>
        </div>
        <button
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-500 disabled:opacity-60"
        >
          Choose CSV file
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
            e.target.value = "";
          }}
        />
        <p className="text-xs text-slate-400">.csv · max 100 MB · 1M rows · 150 columns · comma / semicolon / tab</p>
      </div>

      {samples.length > 0 && (
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
            Or try an example dataset
          </p>
          <div className="grid gap-3 sm:grid-cols-3">
            {samples.map((s) => (
              <button
                key={s.name}
                disabled={busy}
                onClick={() => loadSample(s.name)}
                className="group rounded-xl border border-slate-200 bg-white p-4 text-left transition hover:border-indigo-300 hover:shadow-sm disabled:opacity-50 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-indigo-500/50"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-800 group-hover:text-indigo-600 dark:text-slate-100 dark:group-hover:text-indigo-400">
                    {s.label}
                  </span>
                  <span className="text-xs text-slate-400">
                    {s.rows}×{s.columns}
                  </span>
                </div>
                <p className="mt-1 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                  {s.description}
                </p>
                <span className="mt-2 inline-block rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                  {s.task}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-2.5 text-sm text-rose-700 dark:border-rose-500/30 dark:bg-rose-500/10 dark:text-rose-300">
          {error}
        </div>
      )}
    </div>
  );
}
