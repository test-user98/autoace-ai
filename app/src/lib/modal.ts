/** Thin client for the Modal compute plane. Server-side only. */

export function modalConfig(): { url: string; token: string } {
  const url = process.env.MODAL_ENDPOINT_URL;
  const token = process.env.MODAL_AUTH_TOKEN;
  if (!url || !token) {
    throw new Error("MODAL_ENDPOINT_URL and MODAL_AUTH_TOKEN must be set");
  }
  return { url: url.replace(/\/+$/, ""), token };
}

export async function callModal(path: string, init: RequestInit = {}): Promise<Response> {
  const { url, token } = modalConfig();
  return fetch(`${url}${path}`, {
    ...init,
    headers: { ...(init.headers ?? {}), "x-api-token": token },
    cache: "no-store",
  });
}

/**
 * Only ever hand Modal a URL in our own blob store.
 *
 * Without this, /api/run is an SSRF primitive: any authenticated caller could
 * name an arbitrary host and have the Modal container fetch it.
 */
export function isBlobUrl(candidate: string): boolean {
  try {
    const parsed = new URL(candidate);
    return parsed.protocol === "https:" && parsed.hostname.endsWith(".blob.vercel-storage.com");
  } catch {
    return false;
  }
}
