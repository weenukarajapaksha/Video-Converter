import sys
from pathlib import Path

import pytest

from dubber.errors import DubberError, MediaError
from dubber.stages import lipsync
from dubber.stages.lipsync import LipSyncStage, build_command, check_installation


def test_missing_installation_raises_actionable_error(tmp_path):
    with pytest.raises(DubberError, match="setup_lipsync.ps1") as exc:
        check_installation(tmp_path / "nope" / "python.exe", tmp_path / "Wav2Lip")
    message = str(exc.value)
    assert "python interpreter" in message
    assert "wav2lip_gan.pth" in message and "s3fd.pth" in message


def test_env_vars_override_install_locations(tmp_path, monkeypatch):
    monkeypatch.setenv("DUBBER_LIPSYNC_PYTHON", str(tmp_path / "py.exe"))
    monkeypatch.setenv("DUBBER_LIPSYNC_REPO", str(tmp_path / "repo"))
    assert lipsync.lipsync_python() == tmp_path / "py.exe"
    assert lipsync.lipsync_repo() == tmp_path / "repo"


def test_command_passes_speech_track_fps_and_segments(tmp_path):
    repo = tmp_path / "Wav2Lip"
    cmd = build_command(Path("py"), repo, Path("in.mp4"), Path("voice.wav"),
                        Path("out.mp4"), 29.97, tmp_path / "segs.json")
    assert cmd[1].endswith("wav2lip_runner.py")
    args = dict(zip(cmd[2::2], cmd[3::2]))
    assert args["--audio"] == "voice.wav"
    assert args["--fps"] == "29.97"
    assert args["--segments"] == str(tmp_path / "segs.json")
    assert args["--checkpoint"] == str(repo / "checkpoints" / "wav2lip_gan.pth")


def test_runner_protocol_reports_progress_and_result():
    script = ("print('PROGRESS 0.5000 generating lips'); print('noise'); "
              "print('RESULT {\"frames\": 10}')")
    events = []
    result = LipSyncStage._run_runner([sys.executable, "-c", script],
                                      lambda f, m: events.append((f, m)))
    assert result == {"frames": 10}
    assert events == [(pytest.approx(0.485), "generating lips")]


def test_runner_failure_surfaces_its_output():
    script = "import sys; print('CUDA out of memory'); sys.exit(3)"
    with pytest.raises(MediaError, match="CUDA out of memory"):
        LipSyncStage._run_runner([sys.executable, "-c", script], lambda f, m: None)


@pytest.mark.gpu
def test_wav2lip_end_to_end_on_sample(tmp_path):
    """Needs scripts/setup_lipsync.ps1 and a talking-head clip at tests/assets/face.mp4."""
    import subprocess

    from dubber import media

    face = Path(__file__).parent / "assets" / "face.mp4"
    python, repo = lipsync.lipsync_python(), lipsync.lipsync_repo()
    try:
        check_installation(python, repo)
    except DubberError:
        pytest.skip("lip sync env not installed")
    if not face.exists():
        pytest.skip("no face sample")

    speech = tmp_path / "speech.wav"
    media.extract_audio(face, speech, sample_rate=16000, channels=1)
    out = tmp_path / "out.mp4"
    meta = media.probe(face)
    subprocess.run(build_command(python, repo, face, speech, out, meta.fps, None), check=True)
    result = media.probe(out)
    assert result.resolution == meta.resolution
    assert result.video_duration_s == pytest.approx(meta.video_duration_s, abs=0.1)
