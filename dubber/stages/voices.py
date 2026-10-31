"""Give every diarized speaker a consistent, gender-appropriate voice.

Gender is estimated from the speaker's median fundamental frequency (pYIN pitch
tracking on the clean vocal stem). Typical adult F0: male ~85-155 Hz, female
~165-255 Hz. Each speaker gets a distinct voice; the mapping is saved to
voices.json and can be overridden in the UI before dubbing.
"""

from __future__ import annotations

import numpy as np

from dubber.artifacts import TRANSCRIPT, VOCALS, VOICES
from dubber.config import get_settings
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import Gender, SpeakerProfile, Transcript, VoiceMap
from dubber.stages.transcribe import SR, load_mono_16k

GENDER_SPLIT_HZ = 160.0
MAX_ANALYSIS_S = 40.0  # per speaker; plenty for a stable median


def classify_gender(median_f0: float | None) -> Gender:
    if median_f0 is None or not np.isfinite(median_f0):
        return "unknown"
    return "male" if median_f0 < GENDER_SPLIT_HZ else "female"


def median_f0(audio: np.ndarray, spans: list[tuple[float, float]]) -> float | None:
    import librosa

    pieces, total = [], 0.0
    for start, end in spans:
        if total >= MAX_ANALYSIS_S:
            break
        pieces.append(audio[int(start * SR):int(end * SR)])
        total += end - start
    if not pieces:
        return None
    clip = np.concatenate(pieces)
    if len(clip) < SR // 2:
        return None
    f0, voiced, _ = librosa.pyin(clip, fmin=65, fmax=400, sr=SR, frame_length=1024)
    voiced_f0 = f0[voiced & np.isfinite(f0)]
    return float(np.median(voiced_f0)) if voiced_f0.size else None


def pick_voices(profiles: list[tuple[str, Gender, float]], provider: str,
                f0s: dict[str, float | None] | None = None,
                language: str = "English") -> dict[str, str]:
    """Assign distinct catalog voices; the most talkative speakers get first pick.

    Among unused voices of the speaker's gender, the one whose pitch is closest to
    the speaker's (in log-frequency, i.e. musical distance) wins, so a deep-voiced
    speaker gets a deep English voice. Without a pitch, catalog order decides.
    """
    from dubber.voices_catalog import catalog_for

    catalog = catalog_for(provider, language)  # native voices for the target language
    f0s = f0s or {}
    used: set[str] = set()
    chosen: dict[str, str] = {}
    for speaker, gender, _ in sorted(profiles, key=lambda p: (-p[2], p[0])):
        wanted = gender if gender != "unknown" else "male"
        pool = ([v for v in catalog if v["gender"] == wanted and v["id"] not in used]
                or [v for v in catalog if v["id"] not in used]
                or list(catalog))
        target = f0s.get(speaker)
        if target and target > 0:
            pool = sorted(pool, key=lambda v: abs(np.log(v["f0_hz"] / target)))
        chosen[speaker] = pool[0]["id"]
        used.add(pool[0]["id"])
    return chosen


def voice_f0(provider: str, voice: str) -> float | None:
    from dubber.voices_catalog import VOICES as CATALOG

    return next((v["f0_hz"] for v in CATALOG.get(provider, []) if v["id"] == voice), None)


class VoicesStage(Stage):
    name = "voices"
    outputs = (VOICES,)
    weight = 0.3

    def run(self, job: Job, progress: ProgressFn) -> None:
        provider = job.option("tts_provider", get_settings().tts_provider)
        transcript = job.read_model(TRANSCRIPT, Transcript)
        audio = load_mono_16k(job.path(VOCALS))

        stats: list[tuple[str, Gender, float]] = []
        f0s: dict[str, float | None] = {}
        for i, speaker in enumerate(transcript.speakers):
            spans = [(s.start, s.end) for s in transcript.segments if s.speaker == speaker]
            f0s[speaker] = median_f0(audio, spans)
            stats.append((speaker, classify_gender(f0s[speaker]),
                          sum(e - s for s, e in spans)))
            progress((i + 1) / max(1, len(transcript.speakers)), f"analysed {speaker}")

        language = job.option("target_language", get_settings().target_language)
        voices = pick_voices(stats, provider, f0s, language)
        job.write_model(VOICES, VoiceMap(provider=provider, speakers={
            spk: SpeakerProfile(
                speaker=spk, gender=gender, speaking_time_s=round(talk, 2),
                median_f0_hz=round(f0s[spk], 1) if f0s[spk] else None, voice=voices[spk],
            ) for spk, gender, talk in stats
        }))
        # Casting rationale for the UI/report (VoiceMap has no per-voice pitch field).
        job.write_json("voice_casting.json", {spk: {
            "speaker_f0_hz": round(f0s[spk], 1) if f0s[spk] else None,
            "voice": voices[spk], "voice_f0_hz": voice_f0(provider, voices[spk]),
        } for spk, _, _ in stats})
