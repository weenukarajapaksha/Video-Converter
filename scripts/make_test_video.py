"""Build a synthetic two-speaker Mandarin test video with background music.

Ground truth is known, which makes it useful for regression-testing the whole
pipeline (speaker count, gender detection, silence handling, music retention).

    python scripts/make_test_video.py samples/two_speakers_music.mp4
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

import edge_tts

MALE, FEMALE = "zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"

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


async def _tts(voice: str, text: str, out: Path) -> None:
    await edge_tts.Communicate(text, voice).save(str(out))


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def main(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts, t = [], 1.5
        concat = tmp / "concat.txt"
        ff("-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1.5", str(tmp / "lead.wav"))
        lines = ["file 'lead.wav'"]
        for i, (voice, text, pause) in enumerate(DIALOGUE):
            mp3 = tmp / f"{i}.mp3"
            asyncio.run(_tts(voice, text, mp3))
            wav = tmp / f"{i}.wav"
            ff("-i", str(mp3), "-ar", "24000", "-ac", "1", str(wav))
            ff("-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(pause),
               str(tmp / f"p{i}.wav"))
            lines += [f"file '{i}.wav'", f"file 'p{i}.wav'"]
            parts.append((voice, text))
        concat.write_text("\n".join(lines), encoding="utf-8")
        ff("-f", "concat", "-safe", "0", "-i", str(concat), str(tmp / "speech.wav"))

        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
             "default=nw=1:nk=1", str(tmp / "speech.wav")],
            capture_output=True, text=True, check=True).stdout) + 1.5
        dur = max(dur, 32.0)

        # Soft synthetic "music": a slow arpeggiated chord bed with tremolo.
        music = ("aevalsrc='0.05*sin(2*PI*220*t)*(1+0.3*sin(2*PI*0.5*t))"
                 "+0.04*sin(2*PI*277.18*t)*(1+0.3*sin(2*PI*0.75*t))"
                 "+0.04*sin(2*PI*329.63*t)*(1+0.3*sin(2*PI*1.0*t))"
                 "+0.03*sin(2*PI*110*t)':s=44100:c=stereo")
        ff("-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
           "-f", "lavfi", "-i", music, "-i", str(tmp / "speech.wav"),
           "-filter_complex", "[2:a]aresample=44100,pan=stereo|c0=c0|c1=c0[s];"
                              "[1:a][s]amix=inputs=2:normalize=0[a]",
           "-map", "0:v", "-map", "[a]", "-t", f"{dur:.2f}",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", str(out))
    print(f"wrote {out} ({dur:.1f}s)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples/two_speakers_music.mp4"))
