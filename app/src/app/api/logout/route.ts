import { NextResponse, type NextRequest } from "next/server";

import { COOKIE_NAME, cookieOptions } from "@/lib/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  const response = NextResponse.redirect(new URL("/login", request.url), 303);
  response.cookies.set(COOKIE_NAME, "", cookieOptions(0));
  return response;
}
