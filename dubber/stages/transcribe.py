"""Mandarin ASR: faster-whisper large-v3 (CTranslate2, int8) with batched, VAD-chunked
decoding and a coverage self-check.

Accuracy and latency measures (each verified on real data, see scripts/evaluate.py):
* ASR runs on the Demucs vocal stem, not the mix, so music doesn't cause errors.
* Batched VAD-chunked decoding: speech is cut at silences into short chunks that are
  decoded independently and several at a time on the GPU. Sequential long-form
  decoding can emit an early end-timestamp and silently skip most of a 30 s window
  (it lost 25 s of dialogue on a real test clip); chunking prevents that and is
  ~3x faster.
* Coverage check: VAD speech regions that the transcript doesn't cover are
  re-transcribed on their own, so no dialogue is dropped silently.
* The prompt is a short natural *example* of punctuated simplified Chinese. Whisper
  treats the prompt as preceding transcript, not as an instruction: an instruction
  prompt was echoed back as a hallucinated segment.
* Prompt echoes, subtitle-credit hallucinations and repetition loops are dropped;
  text is normalized to simplified characters and full-width punctuation.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import soundfile as sf

from dubber import ml
from dubber.align import uncovered_speech
from dubber.artifacts import TRANSCRIPT, VOCALS
from dubber.config import get_settings
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import Segment, Transcript, Word

log = logging.getLogger(__name__)

RAW_TRANSCRIPT = "transcript_raw.json"  # pre-diarization ASR output
SR = 16000  # Whisper, pyannote and pitch analysis all work at 16 kHz mono

EXAMPLE_PROMPT = "嗯，好的。那我们开始吧。"
VAD = {"min_silence_duration_ms": 400, "speech_pad_ms": 150}
CHUNK_S = 15  # shorter chunks: less to lose if a chunk is decoded incompletely
BATCH_SIZE = 8  # fits large-v3 int8 in 4 GB VRAM

# Well-known Whisper hallucinations on silence/music (subtitle credits, outros).
HALLUCINATIONS = re.compile(
    r"(字幕|subtitle|订阅|點贊|点赞|转发|打赏|明镜|感谢观看|謝謝觀看|请不吝|by索兰娅|Amara)",
    re.IGNORECASE,
)
# Kana in forced-Chinese output means Whisper drifted into Japanese: drop the segment
# so the coverage check re-decodes that region on its own.
KANA = re.compile(r"[぀-ヿ]")
_FULL_WIDTH = str.maketrans({",": "，", "?": "？", "!": "！", ":": "：", ";": "；"})


def load_mono_16k(path) -> np.ndarray:
    """Decode with libsndfile and resample ourselves (avoids PyAV version pitfalls)."""
    import librosa

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=SR)


def normalize_zh(text: str) -> str:
    """Simplified characters + full-width punctuation (what the translator expects)."""
    import zhconv

    return zhconv.convert(text, "zh-cn").translate(_FULL_WIDTH)


def is_hallucination(text: str, avg_logprob: float, no_speech_prob: float,
                     prompt: str = EXAMPLE_PROMPT) -> bool:
    core = re.sub(r"[\s，。？！,.?!]", "", text)
    if not core:
        return True
    if core in re.sub(r"[\s，。？！,.?!]", "", prompt):  # prompt echoed back
        return True
    if HALLUCINATIONS.search(text) or KANA.search(text):
        return True
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        return True
    return len(core) > 8 and len(set(core)) <= 2  # "哈哈哈哈哈哈哈哈哈" loops


def _to_segment(seg, offset: float = 0.0) -> Segment | None:
    # Whisper marks dialogue turns with a leading "- " in short clips; drop it.
    text = normalize_zh(seg.text.strip()).lstrip("-– ").strip()
    if is_hallucination(text, seg.avg_logprob, seg.no_speech_prob):
        return None
    words = [Word(start=round(w.start + offset, 3), end=round(w.end + offset, 3),
                  text=normalize_zh(w.word), probability=round(w.probability, 3))
             for w in seg.words or []]
    probs = [w.probability for w in words]
    return Segment(id=0, start=round(seg.start + offset, 3), end=round(seg.end + offset, 3),
                   text=text, words=words,
                   confidence=round(sum(probs) / len(probs), 3) if probs else None)


class TranscribeStage(Stage):
    name = "transcribe"
    outputs = (RAW_TRANSCRIPT,)
    weight = 2.0

    def run(self, job: Job, progress: ProgressFn) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        settings = get_settings()
        dev = ml.device()  # imports torch first: loads the CUDA DLLs CTranslate2 needs
        audio = load_mono_16k(job.path(VOCALS))
        duration = len(audio) / SR
        common = dict(language="zh", beam_size=5, word_timestamps=True,
                      condition_on_previous_text=False, initial_prompt=EXAMPLE_PROMPT)

        with ml.gpu_scope():
            progress(0.02, f"loading whisper {settings.whisper_model}")
            model = WhisperModel(settings.whisper_model, device=dev,
                                 compute_type="int8_float16" if dev == "cuda" else "int8")
            if settings.asr_batched:  # fastest; chunks lose cross-chunk context
                progress(0.15, "batched transcription")
                seg_iter, _ = BatchedInferencePipeline(model).transcribe(
                    audio, batch_size=BATCH_SIZE, vad_filter=True, vad_parameters=VAD,
                    chunk_length=CHUNK_S, **common)
            else:  # most accurate; skipped windows are caught by the coverage check
                progress(0.15, "transcribing")
                seg_iter, _ = model.transcribe(audio, vad_filter=True, vad_parameters=VAD,
                                               **common)
            segments = []
            for seg in seg_iter:
                progress(0.15 + 0.65 * min(1.0, seg.end / duration), f"{seg.end:.0f}s")
                if s := _to_segment(seg):
                    segments.append(s)

            # Coverage self-check: re-decode speech the first pass left uncovered.
            speech = [(r["start"] / SR, r["end"] / SR)
                      for r in get_speech_timestamps(audio, VadOptions(**VAD))]
            holes = uncovered_speech(speech, segments)
            for i, (start, end) in enumerate(holes):
                progress(0.8 + 0.15 * i / len(holes), f"recovering {start:.1f}-{end:.1f}s")
                clip = audio[int(start * SR):int(end * SR)]
                recovered, _ = model.transcribe(clip, vad_filter=False, **common)
                segments += [s for s in (_to_segment(r, start) for r in recovered) if s]
            if holes:
                log.warning("re-transcribed %d uncovered speech region(s)", len(holes))
            del model

        segments.sort(key=lambda s: s.start)
        segments = [s.model_copy(update={"id": i}) for i, s in enumerate(segments)]
        speech_s = sum(e - s for s, e in speech)
        job.write_model(RAW_TRANSCRIPT, Transcript(
            segments=segments, speech_ratio=round(speech_s / duration, 3) if duration else None,
        ))
        job.write_json("asr_stats.json", {
            "vad_speech_regions": len(speech), "recovered_regions": len(holes),
            "segments": len(segments),
            "low_confidence_segments": [s.id for s in segments
                                        if s.confidence is not None and s.confidence < 0.6],
        })
        progress(1.0, f"{len(segments)} segments"
                      + (f", {len(holes)} region(s) recovered" if holes else ""))

    @staticmethod
    def raw(job: Job) -> Transcript:
        return job.read_model(RAW_TRANSCRIPT, Transcript)


__all__ = ["TranscribeStage", "TRANSCRIPT"]
