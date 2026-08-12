"use client";

import { upload } from "@vercel/blob/client";
import { useCallback, useEffect, useRef, useState } from "react";

import type { RunEnvelope, StatusResponse } from "@/lib/types";

import ResultsTable from "./ResultsTable";
import ValidationReport from "./ValidationReport";

type Phase = "idle" | "uploading" | "running" | "done" | "error";

const POLL_MS = 2000;

export default function BatchRunner() {
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [uploadPct, setUploadPct] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<RunEnvelope | null>(null);

  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  // setInterval does not serialize async callbacks: a poll slower than POLL_MS
  // would overlap the next one, and two polls both seeing "done" would each try
  // to delete the blob.
  const polling = useRef(false);

  const stopTimer = useCallback(() => {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
  }, []);

  useEffect(() => stopTimer, [stopTimer]);

  async function start() {
    if (!file) return;
    setPhase("uploading");
    setUploadPct(0);
    setElapsed(0);
    setMessage("");
    setResult(null);

    let blobUrl: string;
    try {
      // Direct browser -> Vercel Blob. The ZIP never passes through a serverless
      // function: the body limit there is ~4.5 MB and one provided clip alone is
      // 2.8 MB, so the function path cannot work for a batch.
      const blob = await upload(file.name, file, {
        access: "private",
        handleUploadUrl: "/api/blob/upload",
        multipart: true,
        contentType: file.type || "application/zip",
        onUploadProgress: ({ percentage }) => setUploadPct(Math.round(percentage)),
      });
      blobUrl = blob.url;
    } catch (error) {
      setPhase("error");
      setMessage(`Upload failed: ${(error as Error).message}`);
      return;
    }

    let callId: string;
    try {
      const response = await fetch("/api/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ blobUrl }),
      });
      const payload = (await response.json()) as { call_id?: string; error?: string };
      if (!response.ok || !payload.call_id) throw new Error(payload.error ?? "no call id returned");
      callId = payload.call_id;
    } catch (error) {
      setPhase("error");
      setMessage(`Could not start the batch: ${(error as Error).message}`);
      return;
    }

    setPhase("running");
    const startedAt = Date.now();
    stopTimer();
    polling.current = false;
    timer.current = setInterval(async () => {
      setElapsed(Math.round((Date.now() - startedAt) / 1000));
      if (polling.current) return;
      polling.current = true;
      try {
        const query = new URLSearchParams({ callId, blobUrl });
        const response = await fetch(`/api/status?${query.toString()}`);
        const payload = (await response.json()) as StatusResponse;

        if ("status" in payload && payload.status === "running") return;

        stopTimer();
        if ("status" in payload && payload.status === "done") {
          setResult(payload);
          setPhase("done");
          return;
        }
        const detail =
          ("error" in payload && payload.error) || "the batch returned an unexpected response";
        setPhase("error");
        setMessage(`Batch failed: ${detail}`);
      } catch (error) {
        stopTimer();
        setPhase("error");
        setMessage(`Lost contact with the batch: ${(error as Error).message}`);
      } finally {
        polling.current = false;
      }
    }, POLL_MS);
  }

  const busy = phase === "uploading" || phase === "running";

  return (
    <>
      <section className="card">
        <h2>Run a batch</h2>
        <div className="row">
          <input
            type="file"
            accept=".zip,application/zip"
            disabled={busy}
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
          <button type="button" onClick={start} disabled={!file || busy} style={{ marginTop: 0 }}>
            {busy ? "Working…" : "Analyze batch"}
          </button>
        </div>

        {phase === "uploading" ? (
          <>
            <p className="muted" style={{ marginTop: 14 }}>
              Uploading direct to blob storage — {uploadPct}%
            </p>
            <progress value={uploadPct} max={100} />
          </>
        ) : null}

        {phase === "running" ? (
          <>
            <p className="muted" style={{ marginTop: 14 }}>
              Processing on Modal — {elapsed}s elapsed. The whole batch runs in one container, so
              cold start is paid once rather than per file.
            </p>
            <progress />
          </>
        ) : null}

        {phase === "error" ? <p className="error">{message}</p> : null}
      </section>

      {result ? <RunSummary result={result} /> : null}
      {result?.report ? <ValidationReport report={result.report} /> : null}
      {result ? <ResultsTable rows={result.rows} /> : null}
    </>
  );
}

function RunSummary({ result }: { result: RunEnvelope }) {
  return (
    <section className="card">
      <h2>Run</h2>
      <div className="stats">
        <Stat label="Files" value={String(result.summary.total)} />
        <Stat label="Succeeded" value={String(result.summary.succeeded)} />
        <Stat label="Failed" value={String(result.summary.failed)} />
        <Stat label="Audio" value={`${(result.summary.total_audio_s / 60).toFixed(2)} min`} />
        <Stat label="Wall clock" value={`${result.timings.total_s.toFixed(1)} s`} />
        <Stat label="Realtime factor" value={`${result.summary.realtime_factor}×`} />
      </div>

      {/* Measured, never assumed — PLAN.md §1.5 derives the cost model from these. */}
      <p className="muted" style={{ marginTop: 16 }}>
        Stages: download {result.timings.download_s.toFixed(2)}s · extract{" "}
        {result.timings.extract_s.toFixed(2)}s · analyze {result.timings.run_batch_s.toFixed(2)}s ·
        compute {result.timings.gpu} ·{" "}
        {result.timings.first_input_on_container
          ? `cold container (idle ${result.timings.container_idle_before_input_s.toFixed(1)}s before this batch; excludes image pull)`
          : "warm container (no cold start on this batch)"}{" "}
        · system <code>{result.system_version}</code>
      </p>

      <div className="row" style={{ marginTop: 12 }}>
        <DownloadButton
          label="Download CSV"
          content={result.csv}
          filename="results.csv"
          mime="text/csv"
        />
        <DownloadButton
          label="Download JSON"
          content={result.json}
          filename="results.json"
          mime="application/json"
        />
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

/** Serves the runner's own `to_csv()` / `to_json()` output verbatim. Nothing is
 * re-derived in TypeScript, which is what keeps the download identical to what
 * the pipeline produced. */
function DownloadButton({
  label,
  content,
  filename,
  mime,
}: {
  label: string;
  content: string;
  filename: string;
  mime: string;
}) {
  function save() {
    const url = URL.createObjectURL(new Blob([content], { type: mime }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  }
  return (
    <button className="secondary" type="button" onClick={save} style={{ marginTop: 0 }}>
      {label}
    </button>
  );
}
