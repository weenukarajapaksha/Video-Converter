import subprocess
from pathlib import Path

import pytest


def make_video(path: Path, seconds: float, *, audio: bool = True, size: str = "640x360",
               rate: int = 25) -> Path:
    """Synthesize an H.264/AAC MP4 with ffmpeg's lavfi test sources."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    cmd += ["-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-ac", "2"] if audio else ["-an"]
    subprocess.run([*cmd, str(path)], check=True)
    return path


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    return make_video(tmp_path_factory.mktemp("media") / "sample.mp4", 32, rate=30)


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    from dubber.config import get_settings

    monkeypatch.setenv("DUBBER_JOBS_DIR", str(tmp_path / "jobs"))
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()
