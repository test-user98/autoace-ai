import { NextResponse, type NextRequest } from "next/server";

import { callModal, isBlobUrl } from "@/lib/modal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Hand an already-uploaded batch to Modal and return immediately.
 *
 * Spawn-then-poll, not run-and-wait: a Vercel hobby function times out at 10 s
 * and a real batch runs for minutes (PLAN.md §H.1).
 */
export async function POST(request: NextRequest) {
  const { blobUrl } = (await request.json()) as { blobUrl?: string };

  if (!blobUrl || !isBlobUrl(blobUrl)) {
    return NextResponse.json({ error: "blobUrl must be a Vercel Blob https URL" }, { status: 400 });
  }

  let response: Response;
  try {
    response = await callModal("/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ zip_url: blobUrl }),
    });
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
  return NextResponse.json(JSON.parse(text));
}
