"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import Markdown from "./Markdown";
import { Spinner } from "./ui";

interface Msg {
  role: "user" | "agent";
  text: string;
  tools?: string[];
}

const SUGGESTIONS = [
  "Summarize what you found",
  "Why is that the best model?",
  "Which features matter most?",
  "What does R² mean?",
  "What missing values did you fix?",
];

export default function ChatPanel({ datasetId, initialMessage }: { datasetId: string; initialMessage?: string }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (initialMessage) {
      setMessages([{ role: "agent", text: initialMessage }]);
    }
  }, [initialMessage]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function send(text: string) {
    if (!text.trim() || busy) return;
    setError(null);
    setMessages((m) => [...m, { role: "user", text }]);
    setInput("");
    setBusy(true);
    try {
      const res = await api.chat(datasetId, text);
      setMessages((m) => [...m, { role: "agent", text: res.reply, tools: res.tool_calls }]);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Chat request failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-[560px] flex-col">
      <div ref={scrollRef} className="thin-scroll flex-1 space-y-4 overflow-y-auto pr-1">
        {messages.length === 0 && !busy && (
          <div className="py-10 text-center text-sm text-slate-400">
            Ask the agent about your data…
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                m.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-100 dark:bg-slate-800"
              }`}
            >
              {m.role === "user" ? (
                <p className="text-sm">{m.text}</p>
              ) : (
                <>
                  <Markdown text={m.text} />
                  {m.tools && m.tools.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1 border-t border-slate-200 pt-2 dark:border-slate-700">
                      {m.tools.map((t) => (
                        <span
                          key={t}
                          className="rounded bg-white px-1.5 py-0.5 font-mono text-[10px] text-slate-500 dark:bg-slate-900 dark:text-slate-400"
                        >
                          ⚙ {t}()
                        </span>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}
        {busy && (
          <div className="flex justify-start">
            <div className="rounded-2xl bg-slate-100 px-4 py-3 dark:bg-slate-800">
              <Spinner label="Agent is working…" />
            </div>
          </div>
        )}
        {error && <div className="text-sm text-rose-500">{error}</div>}
      </div>

      {messages.length === 0 && (
        <div className="flex flex-wrap gap-1.5 py-3">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => send(s)}
              className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600 hover:border-indigo-300 hover:text-indigo-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="mt-3 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the analysis, models, metrics…"
          className="flex-1 rounded-lg border border-slate-300 bg-white px-3.5 py-2.5 text-sm outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 dark:border-slate-700 dark:bg-slate-900 dark:focus:ring-indigo-500/20"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}
