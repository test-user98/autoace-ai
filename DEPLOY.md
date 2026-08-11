# DEPLOY — hosted dashboard

Two planes, per `PLAN.md` §1.5 and §H:

| Plane | Host | What runs there |
|---|---|---|
| Control | Vercel (`app/`) | login, direct-to-blob upload, job dispatch, results, downloads |
| Compute | Modal (`modal_app.py`) | zip download → extract → `run_batch()` → JSON envelope |

The dashboard never processes audio. It mints a blob upload token, hands Modal a
URL, and polls. That is what keeps it inside Vercel's ~4.5 MB request-body limit
and 10 s function timeout.

Everything below needs credentials, so none of it has been run. Do the steps in
order — the Vercel side needs the Modal URL that step 2 prints.

---

## 0. Prerequisites

```bash
npm i -g vercel          # Vercel CLI
uv tool install modal    # or: pipx install modal
modal token new          # opens a browser, writes ~/.modal.toml
vercel login
```

Generate the two shared secrets now; you will paste each into two places:

```bash
openssl rand -hex 32     # -> AUTH_SECRET      (Vercel only)
openssl rand -hex 32     # -> MODAL_AUTH_TOKEN (Vercel *and* Modal)
```

---

## 1. Create the Modal secret

The Modal web endpoint is on the public internet. Both its routes require an
`x-api-token` header matching this value; without the secret it would be an open
compute endpoint.

```bash
modal secret create autoace-voice-trial MODAL_AUTH_TOKEN=<the second hex string>
```

The secret name is `SECRET_NAME` in `modal_app.py`. Deploy fails fast if it does
not exist.

---

## 2. Deploy the compute plane

From the repo root (the image mounts `./src`, so the working directory matters):

```bash
modal deploy modal_app.py
```

First deploy builds the image: `ffmpeg` via apt, then `numpy` / `pydantic` /
`fastapi`. Nothing is downloaded at runtime — that is cold-start mitigation 1 in
`PLAN.md` §1.5.

Modal prints a URL for the `web` function, of the form:

```
https://<workspace>--autoace-voice-trial-web.modal.run
```

**Save it — that is `MODAL_ENDPOINT_URL`.** Verify:

```bash
curl -s -H "x-api-token: <MODAL_AUTH_TOKEN>" \
  https://<workspace>--autoace-voice-trial-web.modal.run/health
# {"ok":true,"app":"autoace-voice-trial","gpu":"none (cpu)"}

# and confirm the gate closes:
curl -s -o /dev/null -w '%{http_code}\n' \
  https://<workspace>--autoace-voice-trial-web.modal.run/health    # 401
```

End-to-end check with a real zip, no dashboard involved:

```bash
modal run modal_app.py --zip-path /path/to/batch.zip
```

---

## 3. Create the Vercel project and Blob store

```bash
cd app
vercel link          # create or select a project; root directory = app
```

Then in the Vercel dashboard: **Storage → Create → Blob**, and connect the store
to this project. That injects `BLOB_READ_WRITE_TOKEN` automatically — do not set
it by hand.

> The repo is private and must stay that way (it tracks AutoAce ground truth).
> Vercel deploys from a private repo without any extra configuration.

---

## 4. Set the Vercel environment variables

Project Settings → Environment Variables, for **Production, Preview and
Development**:

| Variable | Value | Notes |
|---|---|---|
| `DASHBOARD_USER` | e.g. `autoace` | the login username |
| `DASHBOARD_PASSWORD` | a long random string | the login password |
| `AUTH_SECRET` | first `openssl rand -hex 32` | HMAC key for the session cookie; ≥ 16 chars or every request 500s |
| `MODAL_ENDPOINT_URL` | URL from step 2 | no trailing slash required |
| `MODAL_AUTH_TOKEN` | second `openssl rand -hex 32` | must equal the value in the Modal secret |
| `BLOB_READ_WRITE_TOKEN` | *(auto)* | injected by connecting the Blob store |

CLI equivalent:

```bash
vercel env add DASHBOARD_USER production
vercel env add DASHBOARD_PASSWORD production
vercel env add AUTH_SECRET production
vercel env add MODAL_ENDPOINT_URL production
vercel env add MODAL_AUTH_TOKEN production
```

---

## 5. Deploy the control plane

```bash
cd app
vercel --prod
```

If you connect the GitHub repo instead of using the CLI, set **Root Directory =
`app`** in Project Settings, otherwise Vercel looks for `package.json` at the
repo root and the build fails.

---

## 6. Verify the deployment

1. Open the production URL → you are redirected to `/login`.
2. `curl -s -o /dev/null -w '%{http_code}\n' https://<app>.vercel.app/api/run -X POST` → **401**
   (the API is gated, not just the pages).
3. Sign in with `DASHBOARD_USER` / `DASHBOARD_PASSWORD`.
4. Upload a batch ZIP. The upload progress bar moves before any request reaches
   a serverless function — that is the direct-to-blob path working.
5. Batch validation, the results table and both download buttons appear.

Record the URL and credentials in `STATE.md` §4 once verified.

---

## Local development

```bash
cd app
npm install
vercel env pull .env.local      # or hand-write it from ../.env.example
npm run dev                      # http://localhost:3000
```

Two caveats when running locally:

- `@vercel/blob`'s `onUploadCompleted` callback does not fire against
  `localhost`. Nothing here depends on it — the client passes the blob URL to
  `/api/run` directly — but do not add a dependency on it later.
- `AUTH_SECRET` must be set or every request throws. `next build` does not need
  it; the running server does.

---

## Environment variables, complete list

**Vercel project** — `DASHBOARD_USER`, `DASHBOARD_PASSWORD`, `AUTH_SECRET`,
`MODAL_ENDPOINT_URL`, `MODAL_AUTH_TOKEN`, `BLOB_READ_WRITE_TOKEN` (auto).

**Modal secret `autoace-voice-trial`** — `MODAL_AUTH_TOKEN`. Phase 2 adds
`HF_TOKEN` to the same secret for the gated pyannote repos; the CPU pipeline
shipping today does not need it.

No secret is committed. `.env.example` documents every variable with no values.

---

## Data handling

- Uploaded blobs get `addRandomSuffix: true`, so URLs are unguessable.
- `/api/status` calls `del(blobUrl)` as soon as the batch reaches a terminal
  state — retention is the run itself.
- The Modal container extracts into a temp directory removed in a `finally`
  block, so audio never outlives the invocation.
- **Residual exposure:** between upload and deletion the blob is
  public-read-if-you-know-the-URL. Vercel Blob supports `access: 'private'` with
  a server-minted presigned GET, which would close this; it needs the
  `issueSignedToken` → `presignUrl` pair and could not be verified without a live
  store, so it is deliberately left as a documented upgrade rather than untested
  code.

---

## Phase 2: turning on the GPU

`modal_app.py` has one marked seam. To enable:

1. `GPU = "T4"` (Turing / sm_75 — `float16` or `int8` only, never `bf16`).
2. Add the torch / faster-whisper / SER wheels to the `image` chain.
3. Bake the weights into the same image layer. Never download at runtime.
4. Replace `StubPredictor` in `run_batch_remote` with the real predictor.

Nothing else in either plane changes: the envelope, the dashboard and the
downloads are all keyed off the `Predictor` protocol, not off the stub.
