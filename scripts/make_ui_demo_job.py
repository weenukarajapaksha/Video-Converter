"""Create a fake, fully "completed" job so the web UI can be previewed without the ML stack.

    python scripts/make_ui_demo_job.py            # -> jobs/<id>/ (printed)

Uses samples/two_speakers_music.mp4 (run scripts/make_test_video.py first). The
dubbed video is a stand-in copy of the input; every JSON artifact is built with the
real pydantic schemas, so the UI sees exactly the shapes the pipeline produces.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubber import artifacts as A  # noqa: E402
from dubber.config import get_settings  # noqa: E402
from dubber.media import probe  # noqa: E402
from dubber.pipeline import Job  # noqa: E402
from dubber.pipeline.job import StageRecord, StageStatus, _now  # noqa: E402
from dubber.schemas import (DubTrack, Segment, SpeakerProfile, SynthesizedLine,  # noqa: E402
                            TranslatedSegment, Translation, TranslationIssue,
                            TranslationReview, Transcript, VoiceMap)
from dubber.subtitles import to_srt  # noqa: E402

# (start, end, speaker, zh, en, emotion, final_duration, tempo, rewrites)
LINES = [
    (1.50, 3.20, "SPEAKER_01", "你今天为什么来这里？", "Why are you here today?",
     "serious", 1.42, 1.0, 0),
    (3.85, 8.60, "SPEAKER_02", "我想和你谈谈。上次的事情，我一直觉得很对不起你。",
     "I wanted to talk. I still feel terrible about last time.", "sad", 3.90, 1.0, 0),
    (9.40, 12.90, "SPEAKER_01", "算了吧，都过去了。你也不用这么放在心上。",
     "Forget it, it's over. Don't dwell on it.", "calm", 2.65, 1.0, 0),
    (13.40, 19.10, "SPEAKER_02", "不行，这件事我必须说清楚。俗话说，解铃还须系铃人。",
     "No, I have to clear this up. I made this mess, I'll fix it.", "serious", 3.80, 1.0, 1),
    (23.10, 25.00, "SPEAKER_01", "好吧，那你说，我听着呢。",
     "All right, go ahead. I'm listening.", "calm", 1.85, 1.0, 0),
    (25.70, 32.60, "SPEAKER_02",
     "其实那天我不是故意迟到的，路上出了车祸，手机也没电了，我真的没办法联系你。",
     "I wasn't late on purpose. There was a crash, my phone died, and I couldn't reach you.",
     "worried", 6.95, 1.12, 1),
    (33.40, 37.10, "SPEAKER_01", "你为什么不早点告诉我？我等了你整整两个小时！",
     "Why didn't you tell me sooner? I waited two whole hours!", "angry", 3.55, 1.06, 0),
    (37.70, 41.20, "SPEAKER_02", "对不起，是我的错。下次我一定提前跟你说。",
     "I'm sorry, it's my fault. Next time I'll tell you first.", "sad", 3.30, 1.0, 0),
    (41.80, 46.30, "SPEAKER_01", "行，这次就原谅你。不过请我吃饭可是少不了的哦。",
     "Fine, you're forgiven. But you're definitely buying me dinner.", "happy", 3.40, 1.0, 0),
]

STAGE_TIMES = {"ingest": 1.8, "separate": 38.4, "transcribe": 21.7, "diarize": 9.2,
               "voices": 2.1, "translate": 14.6, "synthesize": 26.3, "mix": 3.4, "render": 2.2}


def main() -> None:
    video = ROOT / "samples" / "two_speakers_music.mp4"
    if not video.exists():
        sys.exit("run scripts/make_test_video.py first")
    job = Job.create(get_settings().jobs_dir, video,
                     {"target_language": "English", "tts_provider": "edge", "review": True,
                      "lipsync": False, "demo": True})
    meta = probe(video)
    job.write_model(A.METADATA, meta)

    segments = [Segment(id=i, start=s, end=e, speaker=spk, text=zh, confidence=0.93)
                for i, (s, e, spk, zh, *_rest) in enumerate(LINES)]
    job.write_model(A.TRANSCRIPT, Transcript(segments=segments, speech_ratio=0.66))

    budgets = []
    for i, (s, e, *_r) in enumerate(LINES):
        nxt = LINES[i + 1][0] - 0.15 if i + 1 < len(LINES) else meta.duration_s
        budgets.append(round(max(e - s, min(nxt, e + 1.5) - s), 3))

    job.write_model(A.TRANSLATION, Translation(
        model="gemini-3.8-flash",
        segments=[TranslatedSegment(id=i, start=s, end=e, speaker=spk, source_text=zh, text=en,
                                    emotion=emo, max_duration_s=b)
                  for i, ((s, e, spk, zh, en, emo, *_r), b) in enumerate(zip(LINES, budgets))],
        review=TranslationReview(
            accuracy=9, naturalness=9, tone=8,
            summary="Faithful, natural dialogue. One literal idiom and one over-long line were "
                    "revised to fit timing.",
            issues=[
                TranslationIssue(id=3, category="idiom",
                                 comment="Literal rendering of 解铃还须系铃人 sounds unnatural.",
                                 before="As they say, whoever tied the bell must untie it.",
                                 after="I made this mess, I'll fix it."),
                TranslationIssue(id=5, category="length",
                                 comment="Too many words for a 7.2 s slot.",
                                 before="Actually I really wasn't late on purpose that day, "
                                        "there was a car accident on the road and my phone "
                                        "also ran out of battery.",
                                 after="I wasn't late on purpose. There was a crash, my phone "
                                       "died, and I couldn't reach you."),
            ]),
    ))

    job.write_model(A.VOICES, VoiceMap(provider="edge", speakers={
        "SPEAKER_01": SpeakerProfile(speaker="SPEAKER_01", gender="female", median_f0_hz=224.6,
                                     speaking_time_s=17.2, voice="en-US-AvaNeural",
                                     style="warm, a little sharp when annoyed"),
        "SPEAKER_02": SpeakerProfile(speaker="SPEAKER_02", gender="male", median_f0_hz=118.3,
                                     speaking_time_s=19.9, voice="en-US-AndrewNeural", style=""),
    }))

    job.write_model(A.DUB_TRACK, DubTrack(lines=[
        SynthesizedLine(id=i, path=f"tts/line_{i:04d}.wav", start=s,
                        raw_duration_s=round(fd * tempo, 3), final_duration_s=fd, tempo=tempo,
                        rewrites=rw, text=en)
        for i, (s, e, spk, zh, en, emo, fd, tempo, rw) in enumerate(LINES)]))

    job.path(A.SUBTITLES_EN).write_text(to_srt((s, s + fd, en) for s, _e, _sp, _zh, en, _em, fd,
                                               _t, _r in LINES), encoding="utf-8")
    job.path(A.SUBTITLES_ZH).write_text(to_srt(((s, e, zh) for s, e, _sp, zh, *_r in LINES),
                                               cjk=True), encoding="utf-8")
    shutil.copy2(job.input_video, job.path(A.OUTPUT_VIDEO))

    t = _now() - timedelta(minutes=5)
    for name, secs in STAGE_TIMES.items():
        job.manifest.stages[name] = StageRecord(
            status=StageStatus.DONE, attempts=2 if name == "translate" else 1, started_at=t,
            finished_at=t + timedelta(seconds=secs), duration_s=secs)
        t += timedelta(seconds=secs)
    job.add_cost("translate", "gemini", "gemini-3.8-flash", 3164, "tokens", 0.00865)
    job.add_cost("synthesize", "tts", "edge tts", 30.9, "audio seconds", 0.0)
    job.add_cost("synthesize", "gemini", "gemini-3.8-flash rewrites", 612, "tokens", 0.0017)
    job.save()
    job.path("api_state.json").write_text(json.dumps(
        {"state": "done", "edited": [], "error": None}), encoding="utf-8")
    print(f"demo job: {job.root}  ->  http://127.0.0.1:8000/#/jobs/{job.id}")


if __name__ == "__main__":
    main()
