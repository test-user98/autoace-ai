/**
 * End-to-end check against the DEPLOYED system.
 *
 * Exercises the exact path an evaluator takes — private blob upload, signed-URL
 * handoff, Modal batch, poll to completion — rather than any local shortcut.
 * Unit tests cannot catch what this catches: every defect found so far in the
 * deployed stack (annotation resolution, decorator misattachment, raw control
 * characters in JSON, the wrong TimeoutError class) passed local checks and
 * only failed against the live URL.
 *
 * Usage:
 *   BLOB_READ_WRITE_TOKEN=... DASHBOARD_URL=... DASHBOARD_USER=... \
 *   DASHBOARD_PASSWORD=... node scripts/e2e_check.mjs path/to/batch.zip
 *
 * Exits non-zero on the first failed assertion.
 */

import { readFileSync } from "node:fs";
import { head, del } from "@vercel/blob";
import { upload } from "@vercel/blob/client";

const BASE = (process.env.DASHBOARD_URL ?? "").replace(/\/+$/, "");
const USER = process.env.DASHBOARD_USER ?? "admin";
const PASS = process.env.DASHBOARD_PASSWORD ?? "admin";
const ZIP = process.argv[2];

if (!BASE || !ZIP) {
  console.error("usage: DASHBOARD_URL=... node scripts/e2e_check.mjs <batch.zip>");
  process.exit(2);
}

let failures = 0;
const check = (label, ok, detail = "") => {
  console.log(`${ok ? "  PASS" : "  FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures += 1;
};

async function main() {
  console.log(`\n=== 1. auth gate (${BASE}) ===`);

  const anonRun = await fetch(`${BASE}/api/run`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
    redirect: "manual",
  });
  check("unauthenticated /api/run is rejected", anonRun.status === 401, `got ${anonRun.status}`);

  const bad = await fetch(`${BASE}/api/login`, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username: USER, password: "definitely-wrong" }),
    redirect: "manual",
  });
  check(
    "wrong password is rejected",
    (bad.headers.get("location") ?? "").includes("error=1"),
    bad.headers.get("location") ?? "",
  );

  const good = await fetch(`${BASE}/api/login`, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username: USER, password: PASS }),
    redirect: "manual",
  });
  const cookie = (good.headers.getSetCookie?.() ?? [])
    .map((c) => c.split(";")[0])
    .join("; ");
  check("correct password issues a session cookie", cookie.includes("session="));
  if (!cookie) throw new Error("no session cookie — cannot continue");

  console.log("\n=== 2. upload via the APP'S OWN route (the path a user takes) ===");

  // Guard against the exact failure this check previously missed: the product
  // uploaded with access:"public" against a private store, so every browser
  // upload hung forever — while this script called put() directly with the
  // read-write token and reported ALL CHECKS PASSED. A check that exercises a
  // path no user takes is worse than no check. Pin the two together.
  const runnerSrc = readFileSync(new URL("../src/app/BatchRunner.tsx", import.meta.url), "utf8");
  const runnerAccess = runnerSrc.match(/access:\s*"(\w+)"/)?.[1];
  check(
    "this check uploads the same way the product does",
    runnerAccess === "private",
    `BatchRunner.tsx uses access:"${runnerAccess}"`,
  );

  const bytes = readFileSync(ZIP);
  let blob;
  try {
    blob = await upload(`batches/e2e-${Date.now()}.zip`, bytes, {
      access: "private",
      handleUploadUrl: `${BASE}/api/blob/upload`,
      headers: { cookie }, // the browser sends this automatically
      contentType: "application/zip",
    });
  } catch (err) {
    check("upload through /api/blob/upload succeeded", false, err.message);
    throw err;
  }
  check("upload through /api/blob/upload succeeded", Boolean(blob.url), `${(bytes.length / 1e6).toFixed(2)} MB`);

  const anonFetch = await fetch(blob.url, { redirect: "manual" });
  check(
    "blob is NOT publicly readable",
    anonFetch.status !== 200,
    `unauthenticated GET returned ${anonFetch.status}`,
  );

  console.log("\n=== 3. run the batch through the dashboard ===");
  const runRes = await fetch(`${BASE}/api/run`, {
    method: "POST",
    headers: { "content-type": "application/json", cookie },
    body: JSON.stringify({ blobUrl: blob.url }),
  });
  const runBody = await runRes.text();
  check("/api/run accepted the batch", runRes.ok, `${runRes.status} ${runBody.slice(0, 200)}`);
  if (!runRes.ok) {
    await del(blob.url).catch(() => {});
    return;
  }

  const { call_id: callId } = JSON.parse(runBody);
  check("returned a call id", Boolean(callId), callId);

  console.log("\n=== 4. poll to completion ===");
  let result = null;
  for (let i = 1; i <= 30; i += 1) {
    const res = await fetch(`${BASE}/api/status?callId=${encodeURIComponent(callId)}`, {
      headers: { cookie },
    });
    const raw = await res.text();
    let parsed;
    try {
      parsed = JSON.parse(raw); // strict: the browser is no more forgiving
    } catch (err) {
      check("status payload is strictly valid JSON", false, err.message);
      break;
    }
    console.log(`  poll ${i}: ${parsed.status}`);
    if (parsed.status !== "running") {
      result = parsed;
      break;
    }
    await new Promise((r) => setTimeout(r, 6000));
  }

  check("batch finished", result?.status === "done", result?.error ?? String(result?.status));

  if (result?.status === "done") {
    console.log("\n=== 5. results ===");
    const rows = result.rows ?? [];
    check("every clip produced a row", rows.length === 3, `${rows.length} rows`);
    check("no row failed", rows.every((r) => !r.error), rows.map((r) => r.error).filter(Boolean).join("; "));
    check("all 9 schema fields present", rows.every((r) => Object.keys(r.prediction ?? {}).length === 9));
    check("ground-truth labels carried through", rows.every((r) => r.expected));
    check("CSV export present", typeof result.csv === "string" && result.csv.split("\n").length >= 4);
    console.log(`  timings: ${JSON.stringify(result.timings)}`);
    for (const r of rows) {
      console.log(`    ${r.name} -> ${r.prediction?.emotional_tone} (expected ${r.expected?.emotional_tone})`);
    }
  }

  console.log("\n=== 6. cleanup ===");
  // An anonymous GET returns 403 on a private blob whether or not it was
  // deleted, so that assertion could never fail. Ask the store directly.
  await del(blob.url);
  let stillThere = true;
  try {
    await head(blob.url);
  } catch {
    stillThere = false;
  }
  check("blob is really gone from the store", !stillThere);
}

main()
  .then(() => {
    console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`}\n`);
    process.exit(failures === 0 ? 0 : 1);
  })
  .catch((err) => {
    console.error("\nFATAL:", err.message);
    process.exit(1);
  });
