"""Format-agnostic audio ingest.

Everything downstream assumes 16 kHz mono float32. The provided clips are 48 kHz
"stereo" opus, but the channels are bit-identical (corr = 1.0000), so the
downmix is lossless here and halves decode and compute. ffmpeg handles format
detection, so wav/mp3/ogg/m4a/flac all enter through this one path.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000

SUPPORTED_SUFFIXES = frozenset(
    {".wav", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".flac", ".aac", ".wma"}
)


class DecodeError(Exception):
    """Raised when a file cannot be decoded. Callers must fail that file only,
    never the batch."""


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # float32, mono, SAMPLE_RATE
    duration_s: float
    source_path: Path

    @property
    def duration_min(self) -> float:
        return self.duration_s / 60.0


def probe(path: Path) -> dict:
    """Return ffprobe format+stream metadata. Raises DecodeError if unreadable."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,format_name",
            "-show_entries", "stream=codec_name,sample_rate,channels,codec_type",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise DecodeError(f"ffprobe failed for {path.name}: {proc.stderr.strip()[:200]}")
    try:
        meta = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise DecodeError(f"ffprobe returned invalid JSON for {path.name}") from exc

    audio_streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise DecodeError(f"{path.name} contains no audio stream")
    return meta


def load(path: Path) -> Audio:
    """Decode any supported container to 16 kHz mono float32."""
    path = Path(path)
    if not path.is_file():
        raise DecodeError(f"{path.name} does not exist")

    probe(path)  # rejects containers with no audio stream before we decode

    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-i", str(path),
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise DecodeError(f"decode failed for {path.name}: {proc.stderr.decode()[:200]}")

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        raise DecodeError(f"{path.name} decoded to zero samples")

    # Decoded length, not the container's declared duration, which is wrong on
    # truncated and some VBR files.
    return Audio(
        samples=samples,
        duration_s=samples.size / SAMPLE_RATE,
        source_path=path,
    )


def channels_are_identical(path: Path, tol: float = 1e-2) -> bool | None:
    """True when a multi-channel file is duplicated mono rather than genuinely
    channel-separated. Returns None for files that are already mono.

    Worth checking per batch: call-centre recordings are often dual-leg (agent on
    one channel, customer on the other), which would make speaker-role assignment
    free. It does not hold for the provided clips.
    """
    meta = probe(path)
    audio = next(s for s in meta["streams"] if s.get("codec_type") == "audio")
    if int(audio.get("channels", 1)) < 2:
        return None

    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", "2", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise DecodeError(f"decode failed for {path.name}")

    stereo = np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2)
    return bool(np.abs(stereo[:, 0] - stereo[:, 1]).max() < tol)
