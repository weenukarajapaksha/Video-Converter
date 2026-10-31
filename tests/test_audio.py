"""Audio DSP helpers: silence trimming, ducking, time stretch, subtitles."""

import numpy as np
import pytest
import soundfile as sf

from dubber import media
from dubber.stages.mix import duck_envelope
from dubber.subtitles import to_srt
from dubber.tts import SR, trim_silence


def tone(seconds, sr=SR, freq=220.0, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_trim_silence_removes_leading_and_trailing_silence():
    audio = np.concatenate([np.zeros(SR), tone(1.0), np.zeros(SR // 2)])
    trimmed = trim_silence(audio)
    assert len(trimmed) / SR == pytest.approx(1.0, abs=0.1)


def test_duck_envelope_lowers_background_only_during_dialogue():
    sr = 1000
    env = duck_envelope(10 * sr, sr, [(2.0, 4.0)], db=-6.0, ramp_s=0.1)
    assert env[0] == pytest.approx(1.0)
    assert env[3 * sr] == pytest.approx(10 ** (-6 / 20), rel=1e-3)
    assert env[8 * sr] == pytest.approx(1.0)
    assert np.all(np.diff(env[1500:2500]) <= 1e-6)  # smooth ramp down, no clicks


def test_time_stretch_shortens_audio_by_tempo(tmp_path):
    src = tmp_path / "in.wav"
    sf.write(src, tone(2.0), SR)
    out = media.time_stretch(src, tmp_path / "out.wav", tempo=1.25, sample_rate=44100)
    info = sf.info(out)
    assert info.samplerate == 44100
    assert info.duration == pytest.approx(2.0 / 1.25, abs=0.05)


def test_srt_clips_overlaps_and_skips_empty_cues():
    srt = to_srt([(1.0, 3.5, "Why did you come here today?"), (3.0, 4.0, "I want to talk."),
                  (5.0, 6.0, "  ")])
    assert "00:00:01,000 --> 00:00:02,999" in srt
    assert "2\n00:00:03,000 --> 00:00:04,000\nI want to talk." in srt
    assert "3\n" not in srt
