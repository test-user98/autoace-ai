"use client";

import type { BatchReport } from "@/lib/types";

/**
 * The batch-validation panel.
 *
 * Every category is rendered whether or not it fired — a check that is only
 * visible when it trips reads as an absent check. This is a separate data source
 * from the results table on purpose: `run_batch` synthesises failed rows for
 * missing audio, unsupported files and name collisions, but *not* for duplicate
 * manifest rows or unparseable labels, so deriving this panel from the rows
 * would silently drop two of the six categories.
 */
export default function ValidationReport({ report }: { report: BatchReport }) {
  const categories: { name: string; entries: string[] }[] = [
    { name: "Missing audio (row, no file)", entries: report.missing_audio },
    { name: "Unmatched audio (file, no row)", entries: report.unmatched_audio },
    { name: "Unsupported file types", entries: report.unsupported },
    { name: "Duplicate manifest rows", entries: report.duplicate_rows },
    { name: "Name collisions (case-only)", entries: report.name_collisions },
    {
      name: "Unparseable labels",
      entries: report.bad_labels.map((bad) => `${bad.name} — ${bad.reason}`),
    },
  ];

  return (
    <section className="card">
      <h2>Batch validation</h2>
      <p className="muted">
        {report.summary}
        {report.labeled ? " · ground-truth labels detected" : " · unlabeled batch"}
      </p>

      {report.errors.map((error) => (
        <div className="banner" key={error}>
          Batch-level problem: {error}
        </div>
      ))}

      <div className="checks">
        {categories.map((category) => (
          <div
            className={`check ${category.entries.length ? "flagged" : "clean"}`}
            key={category.name}
          >
            <div className="name">
              <span>{category.name}</span>
              <span className="count">{category.entries.length}</span>
            </div>
            {category.entries.length ? (
              <ul>
                {category.entries.map((entry) => (
                  <li key={entry}>{entry}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ))}
      </div>
    </section>
  );
}
