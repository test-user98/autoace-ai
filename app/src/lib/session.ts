/**
 * Signed session cookie, Web Crypto only.
 *
 * Web Crypto rather than `node:crypto` because `middleware.ts` runs on the Edge
 * runtime, where the Node builtin is unavailable. The same module therefore
 * works in middleware and in route handlers.
 *
 * The cookie is an HMAC over a payload, not a bare `session=1` flag — the login
 * is an explicitly gated criterion, and an unsigned marker cookie is forgeable
 * by anyone who opens devtools.
 */

export const COOKIE_NAME = "av_session";
const TTL_SECONDS = 60 * 60 * 12;

type SessionPayload = { u: string; e: number };

const encoder = new TextEncoder();

function secret(): string {
  const value = process.env.AUTH_SECRET;
  if (!value || value.length < 16) {
    throw new Error("AUTH_SECRET is unset or too short (need >= 16 chars)");
  }
  return value;
}

function toBase64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(value: string): Uint8Array {
  const padded = value.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded + "=".repeat((4 - (padded.length % 4)) % 4));
  return Uint8Array.from(binary, (c) => c.charCodeAt(0));
}

async function hmacKey(usage: KeyUsage[]): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    encoder.encode(secret()),
    { name: "HMAC", hash: "SHA-256" },
    false,
    usage,
  );
}

export async function createSession(user: string): Promise<string> {
  const payload: SessionPayload = { u: user, e: Math.floor(Date.now() / 1000) + TTL_SECONDS };
  const body = toBase64Url(encoder.encode(JSON.stringify(payload)));
  const key = await hmacKey(["sign"]);
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(body));
  return `${body}.${toBase64Url(new Uint8Array(signature))}`;
}

export async function verifySession(token: string | undefined): Promise<SessionPayload | null> {
  if (!token) return null;
  const [body, signature] = token.split(".");
  if (!body || !signature) return null;

  let valid: boolean;
  try {
    const key = await hmacKey(["verify"]);
    // crypto.subtle.verify compares in constant time.
    valid = await crypto.subtle.verify(
      "HMAC",
      key,
      fromBase64Url(signature) as unknown as BufferSource,
      encoder.encode(body),
    );
  } catch {
    return null;
  }
  if (!valid) return null;

  try {
    const payload = JSON.parse(new TextDecoder().decode(fromBase64Url(body))) as SessionPayload;
    if (typeof payload.e !== "number" || payload.e < Math.floor(Date.now() / 1000)) return null;
    return payload;
  } catch {
    return null;
  }
}

export function cookieOptions(maxAge: number = TTL_SECONDS) {
  return {
    httpOnly: true,
    sameSite: "lax" as const,
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge,
  };
}
