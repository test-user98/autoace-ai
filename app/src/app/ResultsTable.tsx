"use client";

import { SCHEMA_FIELDS, type RowResult } from "@/lib/types";

const HEADERS: Record<string, string> = {
  emotional_tone: "tone",
  emotional_intensity: "intensity",
  background_noise_present: "noise",
  background_noise_type: "noise type",
  background_noise_severity: "severity",
  audio_quality: "quality",
  speaker_overlap_present: "overlap",
  long_silence_present: "long silence",
  confidence: "conf.",
};

function render(value: string | number | boolean | undefined): string {
  if (value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

/**
 * One row per file, all nine schema fields, and failed files kept inline with
 * their error text rather than hidden in a separate list — the brief requires
 * the dashboard to identify which file failed and why.
 */
export default function ResultsTable({ rows }: { rows: RowResult[] }) {
  if (!rows.length) {
    return (
      <section className="card">
        <h2>Results</h2>
        <p className="muted">No files were processed.</p>
      </section>
    );
  }

  return (
    <section className="card">
      <h2>Results ({rows.length})</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>file</th>
              <th>audio (s)</th>
              {SCHEMA_FIELDS.map((field) => (
                <th key={field}>{HEADERS[field]}</th>
              ))}
              <th>error</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${row.name}-${index}`} className={row.error ? "failed" : undefined}>
                <td>{row.name}</td>
                <td>{row.duration_s ? row.duration_s.toFixed(1) : "—"}</td>
                {SCHEMA_FIELDS.map((field) => (
                  <td key={field}>{render(row.prediction?.[field])}</td>
                ))}
                <td className="err">{row.error ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
