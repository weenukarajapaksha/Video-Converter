"""Mandarin ASR with faster-whisper (CTranslate2) + Silero VAD + word timestamps."""

from __future__ import annotations

import re

import numpy as np
import soundfile as sf

from dubber import ml
from dubber.artifacts import TRANSCRIPT, VOCALS
from dubber.config import get_settings
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import Segment, Transcript, Word

RAW_TRANSCRIPT = "transcript_raw.json"  # pre-diarization ASR output
SR = 16000  # Whisper, pyannote and pitch analysis all work at 16 kHz mono


def load_mono_16k(path) -> np.ndarray:
    """Decode with libsndfile and resample ourselves (avoids PyAV version pitfalls)."""
    import librosa

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=SR)

# Priming with simplified-Chinese, punctuated text steers Whisper towards simplified
# characters and proper sentence punctuation (it otherwise often omits both).
INITIAL_PROMPT = "以下是普通话的对话，请使用简体中文并加上标点符号。"

# Well-known Whisper hallucinations on silence/music (subtitle credits, outros).
HALLUCINATIONS = re.compile(
    r"(字幕|subtitle|订阅|點贊|点赞|转发|打赏|明镜|感谢观看|謝謝觀看|请不吝|by索兰娅|Amara)",
    re.IGNORECASE,
)


def is_hallucination(text: str, avg_logprob: float, no_speech_prob: float) -> bool:
    if not text.strip():
        return True
    if HALLUCINATIONS.search(text):
        return True
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        return True
    # e.g. "哈哈哈哈哈哈哈哈哈哈" loops
    return len(text) > 8 and len(set(text)) <= 2


class TranscribeStage(Stage):
    name = "transcribe"
    outputs = (RAW_TRANSCRIPT,)
    weight = 2.0

    def run(self, job: Job, progress: ProgressFn) -> None:
        from faster_whisper import WhisperModel

        settings = get_settings()
        dev = ml.device()
        audio = load_mono_16k(job.path(VOCALS))
        with ml.gpu_scope():
            progress(0.02, f"loading whisper {settings.whisper_model}")
            model = WhisperModel(settings.whisper_model, device=dev,
                                 compute_type="int8_float16" if dev == "cuda" else "int8")
            segments_iter, info = model.transcribe(
                audio,
                language="zh",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 150},
                word_timestamps=True,
                condition_on_previous_text=False,  # avoids repetition loops
                initial_prompt=INITIAL_PROMPT,
            )
            segments: list[Segment] = []
            for seg in segments_iter:
                progress(0.1 + 0.85 * min(1.0, seg.end / info.duration), f"{seg.end:.0f}s")
                text = seg.text.strip()
                if is_hallucination(text, seg.avg_logprob, seg.no_speech_prob):
                    continue
                words = [Word(start=round(w.start, 3), end=round(w.end, 3), text=w.word,
                              probability=round(w.probability, 3)) for w in seg.words or []]
                probs = [w.probability for w in words]
                segments.append(Segment(
                    id=len(segments), start=round(seg.start, 3), end=round(seg.end, 3),
                    text=text, words=words,
                    confidence=round(sum(probs) / len(probs), 3) if probs else None,
                ))
            del model

        speech = sum(s.duration for s in segments)
        job.write_model(RAW_TRANSCRIPT, Transcript(
            segments=segments,
            speech_ratio=round(speech / info.duration, 3) if info.duration else None,
        ))
        progress(1.0, f"{len(segments)} segments, {speech:.0f}s of speech")

    @staticmethod
    def raw(job: Job) -> Transcript:
        return job.read_model(RAW_TRANSCRIPT, Transcript)


__all__ = ["TranscribeStage", "TRANSCRIPT"]
