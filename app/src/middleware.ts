import { NextResponse, type NextRequest } from "next/server";

import { COOKIE_NAME, verifySession } from "@/lib/session";

/**
 * Gate everything except the login form itself.
 *
 * `/api/*` is covered deliberately: gating pages while leaving the API open is
 * the standard hole, and `/api/blob/upload` mints Vercel Blob upload tokens —
 * unauthenticated, anyone could write to the store.
 */
const PUBLIC_PATHS = new Set(["/login", "/api/login"]);

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  if (PUBLIC_PATHS.has(pathname)) return NextResponse.next();

  const session = await verifySession(request.cookies.get(COOKIE_NAME)?.value);
  if (session) return NextResponse.next();

  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const url = request.nextUrl.clone();
  url.pathname = "/login";
  url.search = "";
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
