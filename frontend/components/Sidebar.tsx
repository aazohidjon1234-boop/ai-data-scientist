"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useTheme } from "@/lib/theme";
import { api } from "@/lib/api";

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-lg text-white shadow-sm">
        🤖
      </div>
      <div className="leading-tight">
        <div className="text-sm font-bold tracking-tight text-white">AI Data Scientist</div>
        <div className="text-[11px] font-medium text-indigo-300">Agent</div>
      </div>
    </div>
  );
}

export default function Sidebar() {
  const pathname = usePathname();
  const { theme, toggle } = useTheme();
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .health()
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false));
  }, []);

  const nav = [
    { href: "/", label: "Dashboard", icon: "▦", active: pathname === "/" },
    { href: "/#analyses", label: "Recent analyses", icon: "◷", active: false, anchor: true },
  ];

  return (
    <>
      {/* mobile top bar */}
      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-slate-200 bg-white/90 px-4 py-3 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90 lg:hidden">
        <Logo />
        <button
          onClick={toggle}
          className="rounded-lg border border-slate-200 px-2.5 py-1.5 text-sm text-slate-600 dark:border-slate-700 dark:text-slate-300"
          aria-label="Toggle theme"
        >
          {theme === "light" ? "🌙" : "☀️"}
        </button>
      </header>

      {/* desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 flex-col bg-slate-900 dark:bg-slate-950 dark:ring-1 dark:ring-inset dark:ring-slate-800 lg:flex">
        <div className="px-5 py-5">
          <Link href="/">
            <Logo />
          </Link>
        </div>

        <nav className="flex-1 space-y-1 px-3">
          {nav.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition ${
                item.active
                  ? "bg-indigo-600/20 text-indigo-200"
                  : "text-slate-300 hover:bg-slate-800 hover:text-white"
              }`}
            >
              <span className="w-4 text-center opacity-70">{item.icon}</span>
              {item.label}
            </Link>
          ))}

          <div className="pt-5">
            <div className="px-3 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              About
            </div>
            <p className="px-3 text-xs leading-relaxed text-slate-400">
              The agent profiles your CSV, detects the task, cleans the data, trains and compares real
              scikit-learn models, and explains every result in plain language.
            </p>
          </div>
        </nav>

        <div className="space-y-3 border-t border-slate-800 px-5 py-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${
                  apiOk === null ? "bg-slate-500" : apiOk ? "bg-emerald-400" : "bg-rose-400"
                }`}
              />
              API {apiOk === null ? "checking…" : apiOk ? "connected" : "offline"}
            </span>
            <button onClick={toggle} className="hover:text-white" aria-label="Toggle theme">
              {theme === "light" ? "🌙" : "☀️"}
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
