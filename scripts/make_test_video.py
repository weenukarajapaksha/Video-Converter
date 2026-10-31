"""Build a synthetic two-speaker Mandarin test video with background music.

Ground truth is saved next to the video (``<name>_truth/``): the clean speech and
music stems plus the script with speaker, gender and timing for every line. That
makes the whole pipeline measurable (see scripts/evaluate.py): ASR character
error rate, speaker count, gender detection, separation SDR and silence handling.

    python scripts/make_test_video.py samples/two_speakers_music.mp4
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import edge_tts
import numpy as np
import soundfile as sf

MALE, FEMALE, MALE2 = "zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-YunjianNeural"
GENDER = {MALE: "male", FEMALE: "female", MALE2: "male"}
SR = 44100
LEAD_S = 1.5

# (speaker voice, line, pause after in seconds). Includes an idiom, a fast line,
# a long sentence and a long silent gap.
DIALOGUE = [
    (FEMALE, "你今天为什么来这里？", 0.6),
    (MALE, "我想和你谈谈。上次的事情，我一直觉得很对不起你。", 0.8),
    (FEMALE, "算了吧，都过去了。你也不用这么放在心上。", 0.5),
    (MALE, "不行，这件事我必须说清楚。俗话说，解铃还须系铃人。", 4.0),
    (FEMALE, "好吧，那你说，我听着呢。", 0.7),
    (MALE, "其实那天我不是故意迟到的，路上出了车祸，手机也没电了，我真的没办法联系你。", 0.8),
    (FEMALE, "你为什么不早点告诉我？我等了你整整两个小时！", 0.6),
    (MALE, "对不起，是我的错。下次我一定提前跟你说。", 0.6),
    (FEMALE, "行，这次就原谅你。不过请我吃饭可是少不了的哦。", 2.0),
]

# Soft synthetic "music": a slow chord bed with tremolo.
MUSIC = ("aevalsrc='0.05*sin(2*PI*220*t)*(1+0.3*sin(2*PI*0.5*t))"
         "+0.04*sin(2*PI*277.18*t)*(1+0.3*sin(2*PI*0.75*t))"
         "+0.04*sin(2*PI*329.63*t)*(1+0.3*sin(2*PI*1.0*t))"
         "+0.03*sin(2*PI*110*t)':s=44100:c=stereo")


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def tts(voice: str, text: str, out: Path, rate: str | None = None) -> float:
    kwargs = {"rate": rate} if rate else {}
    asyncio.run(edge_tts.Communicate(text, voice, **kwargs).save(str(out.with_suffix(".mp3"))))
    ff("-i", str(out.with_suffix(".mp3")), "-ar", str(SR), "-ac", "1", str(out))
    return sf.info(out).duration


def build(out: Path, dialogue=DIALOGUE, *, music: bool = True, noise_db: float | None = None,
          lead_s: float = LEAD_S, rate: str | None = None, min_s: float = 32.0) -> Path:
    """Render ``dialogue`` to an MP4 plus ground truth in ``<out stem>_truth/``.

    ``noise_db``: add white noise at this level relative to speech RMS (e.g. -20).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    truth = out.with_name(out.stem + "_truth")
    truth.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pieces, script, t = [np.zeros(int(lead_s * SR), np.float32)], [], lead_s
        for i, (voice, text, pause) in enumerate(dialogue):
            wav = tmp / f"{i}.wav"
            dur = tts(voice, text, wav, rate)
            audio, _ = sf.read(wav, dtype="float32")
            pieces += [audio, np.zeros(int(pause * SR), np.float32)]
            script.append({"start": round(t, 3), "end": round(t + dur, 3), "text": text,
                           "speaker": voice, "gender": GENDER[voice]})
            t += len(audio) / SR + pause
        speech = np.concatenate(pieces)
        total = max(len(speech) / SR + lead_s, min_s)
        speech = np.pad(speech, (0, int(total * SR) - len(speech)))
        sf.write(truth / "speech.wav", speech, SR, subtype="PCM_16")

        n = len(speech)
        bed = np.zeros(n, np.float32)
        if music:
            ff("-f", "lavfi", "-i", MUSIC, "-t", f"{total:.3f}", "-ac", "1",
               str(tmp / "music.wav"))
            m, _ = sf.read(tmp / "music.wav", dtype="float32")
            bed[: min(n, len(m))] += m[:n]
        if noise_db is not None:
            active = speech[np.abs(speech) > 1e-3]
            rms = float(np.sqrt(np.mean(active ** 2))) if active.size else 0.1
            rng = np.random.default_rng(0)
            bed += (rng.standard_normal(n) * rms * 10 ** (noise_db / 20)).astype(np.float32)
        sf.write(truth / "music.wav", bed, SR, subtype="PCM_16")

        mix = np.clip(speech + bed, -1, 1)
        sf.write(tmp / "mix.wav", mix, SR, subtype="PCM_16")
        ff("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-i", str(tmp / "mix.wav"),
           "-map", "0:v", "-map", "1:a", "-t", f"{total:.3f}", "-ac", "2",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", str(out))
    (truth / "script.json").write_text(json.dumps(
        {"duration_s": round(total, 3), "lines": script}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"wrote {out} ({total:.1f}s) + ground truth in {truth}")
    return out


def main(out: Path) -> None:
    build(out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples/two_speakers_music.mp4"))
