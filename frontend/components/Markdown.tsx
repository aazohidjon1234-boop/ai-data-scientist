"use client";

import type { ReactNode } from "react";

/**
 * Minimal, dependency-free Markdown renderer covering the subset used by the
 * agent's explanations and reports: headings, bullets, numbered lists,
 * tables, bold, inline code, paragraphs.
 */
function inline(text: string, keyBase: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) {
      nodes.push(
        <strong key={`${keyBase}-b${i}`} className="font-semibold text-slate-900 dark:text-white">
          {tok.slice(2, -2)}
        </strong>
      );
    } else {
      nodes.push(
        <code key={`${keyBase}-c${i}`} className="rounded bg-slate-100 px-1 py-0.5 text-[12px] text-indigo-700 dark:bg-slate-800 dark:text-indigo-300">
          {tok.slice(1, -1)}
        </code>
      );
    }
    last = m.index + tok.length;
    i++;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function Table({ rows }: { rows: string[] }) {
  const parse = (r: string) =>
    r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const header = parse(rows[0]);
  const body = rows.slice(2).map(parse); // skip the |---| separator line
  return (
    <div className="thin-scroll overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="bg-slate-50 text-left dark:bg-slate-800/60">
            {header.map((h, i) => (
              <th key={i} className="whitespace-nowrap px-3 py-2 font-semibold text-slate-700 dark:text-slate-200">
                {inline(h, `h${i}`)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, ri) => (
            <tr key={ri} className="border-t border-slate-100 dark:border-slate-800">
              {row.map((c, ci) => (
                <td key={ci} className="whitespace-nowrap px-3 py-1.5 text-slate-600 dark:text-slate-300">
                  {inline(c, `c${ri}-${ci}`)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Markdown({ text }: { text: string }) {
  const lines = text.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i++;
      continue;
    }

    // table block
    if (line.trim().startsWith("|")) {
      const rows: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(lines[i].trim());
        i++;
      }
      blocks.push(<Table key={key++} rows={rows} />);
      continue;
    }

    // heading
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      const cls =
        level === 1
          ? "text-xl font-bold"
          : level === 2
            ? "text-lg font-bold"
            : "text-base font-semibold";
      blocks.push(
        <div key={key++} className={`pt-2 ${cls} text-slate-900 dark:text-white`}>
          {inline(h[2], `h${key}`)}
        </div>
      );
      i++;
      continue;
    }

    // bullet list
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      blocks.push(
        <ul key={key++} className="list-disc space-y-1 pl-5 text-sm text-slate-700 dark:text-slate-300">
          {items.map((it, ii) => (
            <li key={ii}>{inline(it, `li${ii}`)}</li>
          ))}
        </ul>
      );
      continue;
    }

    // numbered list
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i++;
      }
      blocks.push(
        <ol key={key++} className="list-decimal space-y-1 pl-5 text-sm text-slate-700 dark:text-slate-300">
          {items.map((it, ii) => (
            <li key={ii}>{inline(it, `ol${ii}`)}</li>
          ))}
        </ol>
      );
      continue;
    }

    // paragraph (merge consecutive plain lines)
    const para: string[] = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith("|") &&
      !/^#{1,4}\s/.test(lines[i]) &&
      !/^\s*[-*]\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    blocks.push(
      <p key={key++} className="text-sm leading-relaxed text-slate-700 dark:text-slate-300">
        {inline(para.join(" "), `p${key}`)}
      </p>
    );
  }

  return <div className="space-y-3">{blocks}</div>;
}
