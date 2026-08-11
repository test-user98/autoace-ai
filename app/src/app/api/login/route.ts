import { NextResponse, type NextRequest } from "next/server";

import { COOKIE_NAME, cookieOptions, createSession } from "@/lib/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Length-independent comparison, so a wrong password leaks no timing signal. */
function constantTimeEqual(a: string, b: string): boolean {
  const encoder = new TextEncoder();
  const left = encoder.encode(a);
  const right = encoder.encode(b);
  let diff = left.length ^ right.length;
  for (let i = 0; i < Math.max(left.length, right.length); i += 1) {
    diff |= (left[i] ?? 0) ^ (right[i] ?? 0);
  }
  return diff === 0;
}

export async function POST(request: NextRequest) {
  const form = await request.formData();
  const username = String(form.get("username") ?? "");
  const password = String(form.get("password") ?? "");

  const expectedUser = process.env.DASHBOARD_USER ?? "";
  const expectedPassword = process.env.DASHBOARD_PASSWORD ?? "";

  const failure = NextResponse.redirect(new URL("/login?error=1", request.url), 303);
  if (!expectedUser || !expectedPassword) return failure;
  if (!constantTimeEqual(username, expectedUser)) return failure;
  if (!constantTimeEqual(password, expectedPassword)) return failure;

  const response = NextResponse.redirect(new URL("/", request.url), 303);
  response.cookies.set(COOKIE_NAME, await createSession(username), cookieOptions());
  return response;
}
