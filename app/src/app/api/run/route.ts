import { issueSignedToken, presignUrl } from "@vercel/blob";
import { NextResponse, type NextRequest } from "next/server";

import { callModal, isBlobUrl } from "@/lib/modal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * How long Modal has to fetch the batch. Long enough for a cold container to
 * start and download, short enough that a leaked URL is not a standing hole.
 */
const READ_URL_TTL_MS = 15 * 60 * 1000;

/**
 * Mint a short-lived signed GET URL for a private blob.
 *
 * The store is private on purpose — the payload is confidential production call
 * audio, and the brief forbids putting it on unapproved public services. A
 * private blob rejects Modal's unauthenticated fetch, so the read capability has
 * to be delegated explicitly and narrowly: this token is scoped to one pathname,
 * one operation, and a 15-minute window.
 */
async function signedReadUrl(blobUrl: string): Promise<string> {
  const pathname = decodeURIComponent(new URL(blobUrl).pathname).replace(/^\/+/, "");
  const validUntil = Date.now() + READ_URL_TTL_MS;

  const token = await issueSignedToken({
    pathname,
    operations: ["get"],
    validUntil,
  });

  const { presignedUrl } = await presignUrl(token, {
    operation: "get",
    pathname,
    access: "private",
    validUntil,
    // Read from origin: the blob was written seconds ago and a CDN edge may not
    // have it yet, which would hand Modal a 404 on a file that exists.
    useCache: false,
  });

  return presignedUrl;
}

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

  let zipUrl: string;
  try {
    zipUrl = await signedReadUrl(blobUrl);
  } catch (error) {
    return NextResponse.json(
      { error: `cannot sign blob read URL: ${(error as Error).message}` },
      { status: 502 },
    );
  }

  let response: Response;
  try {
    response = await callModal("/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ zip_url: zipUrl }),
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
