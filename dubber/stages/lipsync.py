"""Bonus A: re-animate the speaker's mouth to match the English dub (Wav2Lip-GAN).

Wav2Lip's dependency set is incompatible with the main environment, so inference
runs out of process in ``.venv-lipsync`` (see ``scripts/setup_lipsync.ps1``) via
``wav2lip_runner.py``. The runner gets the *speech-only* dub track (music would
confuse the audio encoder) plus the dubbed segment times, so frames without
dialogue are left untouched. The final mux puts the full dub mix back on.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

from dubber import media
from dubber.artifacts import DUB_MIX, DUB_TRACK, DUB_VOICE, LIPSYNC_VIDEO, OUTPUT_VIDEO
from dubber.errors import DubberError, MediaError
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import DubTrack

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("wav2lip_runner.py")
FRAMES_TMP = "lipsync_frames.mp4"
SEGMENTS_TMP = "lipsync_segments.json"
SETUP_HINT = ("Lip sync is not set up. Run:  powershell -ExecutionPolicy Bypass -File "
              "scripts\\setup_lipsync.ps1  (or set DUBBER_LIPSYNC_PYTHON / "
              "DUBBER_LIPSYNC_REPO to an existing Wav2Lip install).")


def lipsync_python() -> Path:
    default = PROJECT_ROOT / ".venv-lipsync" / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return Path(os.environ.get("DUBBER_LIPSYNC_PYTHON", default))


def lipsync_repo() -> Path:
    return Path(os.environ.get("DUBBER_LIPSYNC_REPO", PROJECT_ROOT / "third_party" / "Wav2Lip"))


def required_files(repo: Path) -> dict[str, Path]:
    return {
        "checkpoint": repo / "checkpoints" / "wav2lip_gan.pth",
        "face detector": repo / "face_detection" / "detection" / "sfd" / "s3fd.pth",
    }


def check_installation(python: Path, repo: Path) -> None:
    missing = [f"python interpreter ({python})"] if not python.exists() else []
    missing += [f"{name} ({path})" for name, path in required_files(repo).items()
                if not path.exists()]
    if missing:
        raise DubberError(f"{SETUP_HINT} Missing: {', '.join(missing)}")


def build_command(python: Path, repo: Path, video: Path, audio: Path, out: Path,
                  fps: float, segments: Path | None) -> list[str]:
    cmd = [str(python), str(RUNNER), "--repo", str(repo),
           "--checkpoint", str(required_files(repo)["checkpoint"]),
           "--video", str(video), "--audio", str(audio), "--out", str(out),
           "--fps", f"{fps}", "--ffmpeg", shutil.which("ffmpeg") or "ffmpeg"]
    if segments is not None:
        cmd += ["--segments", str(segments)]
    return cmd


class LipSyncStage(Stage):
    name = "lipsync"
    outputs = (LIPSYNC_VIDEO,)
    weight = 4.0

    def run(self, job: Job, progress: ProgressFn) -> None:
        python, repo = lipsync_python(), lipsync_repo()
        check_installation(python, repo)

        video = job.path(OUTPUT_VIDEO)
        meta = media.probe(video)
        speech = job.path(DUB_VOICE) if job.path(DUB_VOICE).exists() else job.path(DUB_MIX)

        segments = None
        if job.path(DUB_TRACK).exists():
            track = job.read_model(DUB_TRACK, DubTrack)
            segments = job.write_json(SEGMENTS_TMP, [
                [line.start, line.start + line.final_duration_s] for line in track.lines])

        frames = job.path(FRAMES_TMP)
        cmd = build_command(python, repo, video, speech, frames, meta.fps, segments)
        result = self._run_runner(cmd, progress)
        job.write_json("lipsync_stats.json", result)

        progress(0.98, "muxing audio")
        media.run_ffmpeg("-i", frames, "-i", job.path(DUB_MIX),
                         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                         job.path(LIPSYNC_VIDEO))
        frames.unlink(missing_ok=True)

    @staticmethod
    def _run_runner(cmd: list[str], progress: ProgressFn) -> dict:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        tail: deque[str] = deque(maxlen=30)
        result: dict = {}
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("PROGRESS "):
                _, frac, msg = (line.split(" ", 2) + [""])[:3]
                progress(0.97 * float(frac), msg)
            elif line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
            else:
                tail.append(line)
        if proc.wait() != 0:
            raise MediaError("Wav2Lip runner failed:\n" + "\n".join(tail))
        return result
