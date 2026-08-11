import { handleUpload, type HandleUploadBody } from "@vercel/blob/client";
import { NextResponse, type NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_ZIP_BYTES = 512 * 1024 * 1024;

/**
 * Mints a short-lived client token so the browser uploads **straight to Vercel
 * Blob**, never through a serverless function.
 *
 * This is the single hardest constraint in the deployment (PLAN.md §H.2):
 * Vercel's request-body limit is ~4.5 MB and `call_003.ogg` alone is 2.8 MB, so
 * a zipped batch posted through a function fails instantly. The zip bytes never
 * touch this route — only the token request does.
 *
 * The route sits behind `middleware.ts`, so an unauthenticated caller cannot
 * mint tokens against the store. `onUploadCompleted` is intentionally not wired:
 * it needs a publicly reachable callback and does not fire on localhost, so
 * nothing load-bearing may depend on it. The client hands the returned blob URL
 * to /api/run itself.
 */
export async function POST(request: NextRequest) {
  const body = (await request.json()) as HandleUploadBody;

  try {
    const result = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async () => ({
        allowedContentTypes: [
          "application/zip",
          "application/x-zip-compressed",
          "application/octet-stream",
        ],
        maximumSizeInBytes: MAX_ZIP_BYTES,
        // Unguessable URL. Vercel Blob is public-read by default and this is
        // confidential customer call audio; the run flow deletes the blob as
        // soon as the batch finishes (see /api/status).
        addRandomSuffix: true,
      }),
    });
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 400 });
  }
}
