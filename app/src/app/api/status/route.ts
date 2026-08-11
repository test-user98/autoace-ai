import { del } from "@vercel/blob";
import { NextResponse, type NextRequest } from "next/server";

import { callModal, isBlobUrl } from "@/lib/modal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const callId = request.nextUrl.searchParams.get("callId");
  const blobUrl = request.nextUrl.searchParams.get("blobUrl");
  if (!callId) {
    return NextResponse.json({ error: "callId is required" }, { status: 400 });
  }

  let response: Response;
  try {
    response = await callModal(`/status/${encodeURIComponent(callId)}`);
  } catch (error) {
    return NextResponse.json({ error: `cannot reach Modal: ${(error as Error).message}` }, { status: 502 });
  }

  const text = await response.text();
  if (!response.ok) {
    return NextResponse.json(
      { error: `Modal returned ${response.status}: ${text.slice(0, 400)}` },
      { status: 502 },
    );
  }

  const payload = JSON.parse(text) as { status?: string };

  // The batch is finished, so the uploaded audio has no reason to keep existing.
  // Blob URLs are public-read; this is confidential customer audio and privacy
  // is a scored line item, so the retention window is the run itself.
  if (payload.status && payload.status !== "running" && blobUrl && isBlobUrl(blobUrl)) {
    try {
      await del(blobUrl);
    } catch {
      // Best-effort: a failed cleanup must not hide a completed batch.
    }
  }

  return NextResponse.json(payload);
}
