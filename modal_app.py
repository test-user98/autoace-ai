"""Modal deployment of the batch pipeline — the compute plane of PLAN.md §1.5.

Two functions, one app:

* ``web``  — an always-cheap CPU ASGI container holding the HTTP surface.
              ``POST /run`` spawns a batch and returns a call id; ``GET
              /status/{id}`` polls it. Splitting spawn from poll is what lets the
              Vercel dashboard stay inside its 10 s function timeout (§H.1).
* ``run_batch_remote`` — one invocation processes an **entire batch**. Per PLAN
              §1.5, a ~20 s cold start amortised over a 30-clip batch is noise;
              charged per clip it would dominate everything measured.

Everything the pipeline needs is baked into the image at build time. Nothing is
downloaded at runtime — that is the first of the three §1.5 cold-start
mitigations, and it is also why ``ffmpeg`` is an ``apt_install`` and not a
hopeful assumption (``voicetrial.ingest`` shells out to ``ffmpeg``/``ffprobe``;
without it every row fails with "decode failed").

Deploy:  modal deploy modal_app.py      (see DEPLOY.md)
"""

import hmac
import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import modal

APP_NAME = "autoace-voice-trial"
LOCAL_SRC = Path(__file__).parent / "src"

# Name of the Modal secret holding MODAL_AUTH_TOKEN. A Modal web endpoint is on
# the public internet; unauthenticated it is an open compute endpoint.
SECRET_NAME = "autoace-voice-trial"

# Cap on what we will pull from a blob URL. The three provided clips total
# ~5 MB; 512 MB is generous for a real eval batch and still bounds a hostile zip.
MAX_ZIP_BYTES = 512 * 1024 * 1024
MAX_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024


# --------------------------------------------------------------------------
# GPU SEAM — deliberately not wired. The whole pipeline runs on CPU inside the
# cost ceiling, so a GPU would only make the measured $/audio-min in `timings` a
# fiction. To flip it for a heavier model:
#   1. set GPU = "T4"   (Turing/sm_75 — float16 or int8 only, never bf16)
#   2. add the wheels to `image` below
#   3. bake weights into the image in the same layer — never download at runtime
# --------------------------------------------------------------------------
GPU: str | None = None


image = (
    modal.Image.debian_slim(python_version="3.12")
    # voicetrial.ingest shells out to both binaries. Non-negotiable.
    .apt_install("ffmpeg")
    .pip_install(
        "numpy>=2.0",
        "pydantic>=2.9",
        "fastapi[standard]",
        "scikit-learn>=1.9",
        "joblib>=1.4",
        # pyannote supplies speaker-independent overlap and silence. CPU only:
        # it runs 80-144x realtime, so a GPU would add cost without buying
        # anything the cost model needs.
        "torch>=2.4",
        "torchaudio>=2.4",
        "pyannote.audio>=3.3",
        "transformers>=5.0",
    )
    .env({"PYTHONPATH": "/root/src"})
    # Local source must come last in the chain: it is mounted at runtime, so no
    # image layer may depend on it.
    .add_local_dir(LOCAL_SRC, "/root/src", ignore=["**/__pycache__", "**/*.pyc"])
    # Trained models travel with the deploy — never fetched at runtime.
    .add_local_dir(Path(__file__).parent / "models", "/root/models")
)

app = modal.App(APP_NAME, image=image)
auth_secret = modal.Secret.from_name(SECRET_NAME)


# --------------------------------------------------------------------------
# Container-lifetime instrumentation.
#
# PLAN.md §1.5 wants three cost figures, and (b) needs cold start measured, not
# assumed. `_BOOTED_AT` is set once per container at import; the first input to
# land on a container reports how long that container sat between process start
# and useful work. That is a *lower bound* on true cold start — it excludes
# image pull and sandbox creation, which happen before this module is imported.
# Reported as such rather than as the whole number.
# --------------------------------------------------------------------------
_BOOTED_AT = time.monotonic()
_INVOCATIONS = 0


def _download(url: str) -> bytes:
    """Fetch the batch zip from the blob URL the dashboard minted.

    The URL originates from our own Vercel Blob store and the dashboard
    allowlists its hostname before ever sending it here; this end enforces the
    scheme and the size ceiling so a bad URL cannot become unbounded egress.
    """
    import urllib.request

    if not url.lower().startswith("https://"):
        raise ValueError("zip_url must be https")

    req = urllib.request.Request(url, headers={"User-Agent": "autoace-voice-trial"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - scheme checked
        declared = resp.headers.get("Content-Length")
        if declared and int(declared) > MAX_ZIP_BYTES:
            raise ValueError(f"zip too large: {declared} bytes > {MAX_ZIP_BYTES}")
        data = resp.read(MAX_ZIP_BYTES + 1)
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError(f"zip exceeds {MAX_ZIP_BYTES} bytes")
    return data


def _safe_extract(data: bytes, dest: Path) -> None:
    """Extract a zip we did not create.

    Two guards, both cheap: reject members that escape `dest` (zip-slip), and
    bound the total unpacked size (zip bomb). `parse_batch` already handles the
    wrapper-directory shape a zipped folder produces, so nothing is flattened
    here.
    """
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    total = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            try:
                target = (dest / info.filename).resolve()
            except (OSError, ValueError):
                continue  # unrepresentable name; skip the member, keep the batch
            if not target.is_relative_to(root):
                continue  # zip-slip attempt; skip rather than fail the batch
            total += info.file_size
            if total > MAX_UNPACKED_BYTES:
                raise ValueError("zip expands beyond the unpacked size limit")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
            except OSError:
                # e.g. ENAMETOOLONG on a >255-byte filename, which a real export
                # can produce. Skipping one member costs one file; raising here
                # would fail the whole batch and violate the brief's explicit
                # "a single malformed file should not cause the entire batch to
                # fail". The file simply shows up as unmatched in the report.
                continue


def _report_dict(report) -> dict | None:
    """Serialize a `BatchReport` for the dashboard.

    `RunResult.to_json()` deliberately omits the report, so this is the only path
    by which the six directly-scored validation categories reach the UI. Built by
    hand rather than `asdict()` because `BatchItem.audio_path` and `report.root`
    are `Path` objects and would raise on `json.dumps`. `root` is dropped
    outright — it is a container temp path, meaningless and leaky in a UI.
    """
    if report is None:
        return None
    return {
        "ok": report.ok,
        "labeled": report.labeled,
        "summary": report.summary(),
        "item_count": len(report.items),
        "missing_audio": list(report.missing_audio),
        "unmatched_audio": list(report.unmatched_audio),
        "unsupported": list(report.unsupported),
        "duplicate_rows": list(report.duplicate_rows),
        "name_collisions": list(report.name_collisions),
        "bad_labels": [{"name": n, "reason": r} for n, r in report.bad_labels],
        "errors": list(report.errors),
    }


@app.function(
    gpu=GPU,
    timeout=3600,
    secrets=[auth_secret],
    # Keep the whole batch on one container: cold start amortises across it.
    max_containers=4,
)
def run_batch_remote(zip_url: str | None = None, zip_bytes: bytes | None = None) -> dict:
    """Process one whole batch inside one container invocation."""
    global _INVOCATIONS

    from voicetrial.runner import run_batch

    _INVOCATIONS += 1
    is_first = _INVOCATIONS == 1
    container_idle_s = time.monotonic() - _BOOTED_AT

    started = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix="autoace-batch-"))
    try:
        t0 = time.perf_counter()
        if zip_url:
            data = _download(zip_url)
        elif zip_bytes:
            data = zip_bytes
        else:
            raise ValueError("supply either zip_url or zip_bytes")
        download_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        extracted = workdir / "batch"
        _safe_extract(data, extracted)
        extract_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        # No explicit predictor: run_batch selects AcousticPredictor when the
        # models are present and records WHY if it cannot. Hardcoding the stub
        # here silently overrode all of that.
        result = run_batch(extracted)
        run_s = time.perf_counter() - t0

        return {
            "status": "done",
            "system_version": result.system_version,
            "report": _report_dict(result.report),
            "rows": [asdict(row) for row in result.rows],
            "summary": {
                "total": len(result.rows),
                "succeeded": len(result.succeeded),
                "failed": len(result.failed),
                "total_audio_s": round(result.total_audio_s, 3),
                "total_elapsed_s": round(result.total_elapsed_s, 3),
                "realtime_factor": round(result.realtime_factor, 2),
            },
            "timings": {
                "download_s": round(download_s, 3),
                "extract_s": round(extract_s, 3),
                "run_batch_s": round(run_s, 3),
                "total_s": round(time.perf_counter() - started, 3),
                "zip_bytes": len(data),
                # Lower bound on cold start: container process start -> first
                # input. Excludes image pull and sandbox creation.
                "first_input_on_container": is_first,
                "container_idle_before_input_s": round(container_idle_s, 3),
                "gpu": GPU or "none (cpu)",
            },
            # Emitted by the runner itself so the dashboard's downloads are the
            # runner's own output, never re-derived in TypeScript.
            "csv": result.to_csv(),
            # Surfaces WHY the real predictor did not load. Without this the
            # deployment silently serves stub constants and reports success.
            "predictor_warning": getattr(result, "predictor_warning", None),
            "json": result.to_json(),
        }
    finally:
        # Confidential customer audio: never outlive the invocation on disk.
        shutil.rmtree(workdir, ignore_errors=True)


@app.function(
    timeout=60,
    secrets=[auth_secret],
    # Control-surface only: it spawns and polls, it never processes audio.
    min_containers=0,
)
@modal.asgi_app()
def web():
    from fastapi import Body, FastAPI, HTTPException, Request, Response, UploadFile
    from fastapi import File as FastFile

    api = FastAPI(title="AutoAce voice trial — compute plane")

    def _check(request: Request) -> None:
        expected = os.environ.get("MODAL_AUTH_TOKEN", "")
        supplied = request.headers.get("x-api-token", "")
        if not expected or not hmac.compare_digest(expected, supplied):
            raise HTTPException(status_code=401, detail="unauthorized")

    @api.get("/health")
    def health() -> dict:
        return {"ok": True, "app": APP_NAME, "gpu": GPU or "none (cpu)"}

    @api.post("/run")
    def run(request: Request, payload: dict = Body(...)) -> dict:
        """Primary path: process the batch already sitting in blob storage.

        The dashboard uploads direct-to-blob (Vercel's ~4.5 MB function body
        limit rules out proxying a zip), so it hands us a URL, not bytes.
        """
        _check(request)
        zip_url = (payload or {}).get("zip_url")
        if not isinstance(zip_url, str) or not zip_url.lower().startswith("https://"):
            raise HTTPException(status_code=400, detail="zip_url (https) is required")
        call = run_batch_remote.spawn(zip_url=zip_url)
        return {"call_id": call.object_id, "status": "running"}

    @api.post("/run/upload")
    async def run_upload(request: Request, file: UploadFile = FastFile(...)) -> dict:
        """Convenience branch for local testing with `curl -F`. Not the path the
        dashboard uses — large inputs belong in blob storage."""
        _check(request)
        data = await file.read()
        if len(data) > MAX_ZIP_BYTES:
            raise HTTPException(status_code=413, detail="zip too large")
        call = run_batch_remote.spawn(zip_bytes=data)
        return {"call_id": call.object_id, "status": "running"}

    def _json(payload: dict) -> Response:
        """Serialize explicitly.

        The result envelope embeds `to_csv()` output, which is full of newlines.
        Returning the dict and letting the framework encode it produced raw
        control characters inside the JSON string, which `json.loads` tolerates
        but the browser's strict `JSON.parse` rejects — every successful run
        broke the dashboard. `json.dumps` escapes them, so serialize here and
        leave nothing to the framework.
        """
        return Response(content=json.dumps(payload), media_type="application/json")

    @api.get("/status/{call_id}")
    def status(request: Request, call_id: str) -> Response:
        _check(request)
        try:
            call = modal.FunctionCall.from_id(call_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"unknown call: {exc}") from exc
        try:
            return _json(call.get(timeout=0))
        except Exception as exc:
            name = type(exc).__name__
            # `get(timeout=0)` signals "still running" by RAISING a timeout.
            # Which class it raises has moved between Modal versions (builtin
            # TimeoutError vs modal.exception.TimeoutError vs
            # FunctionTimeoutError), and catching the wrong one made every
            # in-flight poll report a failed run instead of progress — the
            # dashboard would never show a batch completing. Match on the name
            # so this cannot silently regress on a client upgrade.
            if "Timeout" in name:
                return _json({"status": "running", "call_id": call_id})
            if "OutputExpired" in name:
                return _json({"status": "error", "error": "result expired; re-run the batch"})
            # The batch function itself raised. Surface it rather than 500ing —
            # a failed run is information, not an outage.
            return _json({"status": "error", "error": f"{name}: {exc}"})

    return api


@app.local_entrypoint()
def main(zip_path: str) -> None:
    """Smoke-test a local zip against the deployed image:

        modal run modal_app.py --zip-path /tmp/batch.zip
    """
    data = Path(zip_path).read_bytes()
    result = run_batch_remote.remote(zip_bytes=data)
    print(json.dumps({k: v for k, v in result.items() if k not in ("csv", "json")}, indent=2))
