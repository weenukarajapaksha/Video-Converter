"""Tasks 4 + 5a: generate English speech and fit every line to its time slot.

Fitting strategy, least audible first:
1. Synthesize all lines in parallel (cached by content hash).
2. Lines that overrun their budget by >12% are sent back to the LLM in ONE batch
   request to be rephrased shorter, then re-synthesized (max 2 rounds).
3. Remaining small overruns are time-stretched with Rubber Band (pitch and
   formants preserved), capped at 1.25x to avoid chipmunk speech.
4. Each line is placed at its original start time on a silent timeline, so
   pauses between lines and silent sections stay exactly as in the source.

``revoice_line`` re-does steps 1, 3 and 4 for a single line after a human edit
(the user's wording is final, so no LLM rewrite), which takes a few seconds.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from dubber import llm, media
from dubber.align import plan_fit
from dubber.artifacts import DUB_TRACK, DUB_VOICE, TRANSLATION, TTS_DIR, VOICES
from dubber.config import get_settings
from dubber.errors import DubberError
from dubber.pipeline.job import Job
from dubber.pipeline.stage import ProgressFn, Stage
from dubber.schemas import (DubTrack, Segment, SynthesizedLine, TranslatedSegment, Translation,
                            VoiceMap)
from dubber.stages.ingest import IngestStage
from dubber.translate import shorten_lines
from dubber.tts import Speech, TTSProvider, get_provider

OUT_SR = 44100
WORKERS = {"gemini": 3, "edge": 6}
MAX_REWRITES = 2


def speak(tts: TTSProvider, job: Job, seg: TranslatedSegment, text: str, voices: VoiceMap,
          language: str) -> Speech:
    prof = voices.speakers[seg.speaker]
    return tts.synthesize(text, prof.voice, seg.emotion, prof.style, language,
                          job.path(TTS_DIR))


def fit_line(job: Job, seg: TranslatedSegment, sp: Speech, text: str,
             rewrites: int) -> SynthesizedLine:
    """Time-stretch (if needed) and resample one clip; no further LLM rewrites."""
    plan = plan_fit(sp.duration, seg.max_duration_s, rewrites_done=MAX_REWRITES,
                    max_rewrites=MAX_REWRITES)
    fitted = media.time_stretch(sp.path, job.path(TTS_DIR) / f"line_{seg.id:04d}.wav",
                                plan.tempo, OUT_SR)
    return SynthesizedLine(
        id=seg.id, path=fitted.relative_to(job.root).as_posix(), start=seg.start,
        raw_duration_s=round(sp.duration, 3),
        final_duration_s=round(sf.info(fitted).frames / OUT_SR, 3),
        tempo=plan.tempo, rewrites=rewrites, text=text,
    )


def assemble_timeline(job: Job, lines: list[SynthesizedLine]) -> None:
    """Place every fitted clip at its start time on a silent track (dub_voice.wav)."""
    total = IngestStage.metadata(job).duration_s
    n = int(round(total * OUT_SR))
    timeline = np.zeros(n + OUT_SR, dtype=np.float32)
    for line in lines:
        audio, _ = sf.read(job.path(line.path), dtype="float32")
        at = int(round(line.start * OUT_SR))
        end = min(len(timeline), at + len(audio))
        timeline[at:end] += audio[: end - at]
    sf.write(job.path(DUB_VOICE), np.clip(timeline[:n], -1, 1), OUT_SR, subtype="PCM_16")


class SynthesizeStage(Stage):
    name = "synthesize"
    outputs = (DUB_TRACK, DUB_VOICE)
    weight = 3.0
    max_attempts = 3

    def run(self, job: Job, progress: ProgressFn) -> None:
        settings = get_settings()
        translation = job.read_model(TRANSLATION, Translation)
        voices = job.read_model(VOICES, VoiceMap)
        tts = get_provider(voices.provider)
        model = job.option("translation_model", settings.translation_model)
        lines = {s.id: s for s in translation.segments}
        texts = {s.id: s.text for s in translation.segments}
        rewrites = {i: 0 for i in lines}
        tts_cost = 0.0
        usage = llm.Usage()

        def voice(seg_id: int) -> Speech:
            return speak(tts, job, lines[seg_id], texts[seg_id], voices,
                         translation.target_language)

        todo = list(lines)
        speech: dict[int, Speech] = {}
        with ThreadPoolExecutor(WORKERS.get(tts.name, 3)) as pool:
            for round_ in range(MAX_REWRITES + 1):
                label = "voiced" if round_ == 0 else f"re-voiced (round {round_})"
                for n, (i, sp) in enumerate(zip(todo, pool.map(voice, todo)), 1):
                    speech[i] = sp
                    tts_cost += sp.cost_usd
                    progress(0.8 * n / len(todo), f"{label} {n}/{len(todo)}")

                over = [i for i in lines if plan_fit(
                    speech[i].duration, lines[i].max_duration_s, rewrites[i],
                    max_rewrites=MAX_REWRITES).action == "rewrite"]
                if not over:
                    break
                progress(0.8, f"shortening {len(over)} over-long line(s)")
                shorter = shorten_lines(
                    [(_as_segment(lines[i]), texts[i], lines[i].max_duration_s) for i in over],
                    model, translation.target_language, usage)
                for i in over:
                    texts[i] = shorter[i]
                    rewrites[i] += 1
                todo = over

        job.add_cost(self.name, "tts", f"{tts.name} tts", sum(s.duration for s in speech.values()),
                     "audio seconds", tts_cost)
        if usage.input_tokens:
            job.add_cost(self.name, "gemini", f"{model} rewrites",
                         usage.input_tokens + usage.output_tokens, "tokens", usage.cost(model))

        progress(0.85, "time-fitting and placing lines")
        placed = [fit_line(job, seg, speech[i], texts[i], rewrites[i])
                  for i, seg in sorted(lines.items(), key=lambda kv: kv[1].start)]

        # Persist the final (possibly shortened) wording so subtitles match the audio.
        job.write_model(TRANSLATION, translation.model_copy(update={"segments": [
            s.model_copy(update={"text": texts[s.id]}) for s in translation.segments]}))
        assemble_timeline(job, placed)
        job.write_model(DUB_TRACK, DubTrack(lines=placed))
        stretched = sum(1 for p in placed if p.tempo > 1.0)
        rewritten = sum(1 for p in placed if p.rewrites)
        progress(1.0, f"{len(placed)} lines, {rewritten} shortened, {stretched} stretched")


def revoice_line(job: Job, line_id: int, text: str | None = None,
                 emotion: str | None = None) -> SynthesizedLine:
    """Re-synthesize one line after a human edit and rebuild dub_voice.wav.

    The edited wording is used as-is (no LLM rewrite); a small overrun is still
    time-stretched up to the usual cap.
    """
    translation = job.read_model(TRANSLATION, Translation)
    voices = job.read_model(VOICES, VoiceMap)
    track = job.read_model(DUB_TRACK, DubTrack)
    seg = next((s for s in translation.segments if s.id == line_id), None)
    if seg is None:
        raise DubberError(f"unknown line id {line_id}")
    if text is not None:
        if not text.strip():
            raise DubberError("text must not be empty")
        seg.text = text.strip()
    if emotion is not None:
        seg.emotion = emotion

    tts = get_provider(voices.provider)
    sp = speak(tts, job, seg, seg.text, voices, translation.target_language)
    line = fit_line(job, seg, sp, seg.text, rewrites=0)
    if sp.cost_usd:
        job.add_cost("revoice", "tts", f"{tts.name} tts", sp.duration, "audio seconds",
                     sp.cost_usd)
        job.save()

    others = [ln for ln in track.lines if ln.id != line_id]
    lines = sorted([*others, line], key=lambda ln: ln.start)
    job.write_model(TRANSLATION, translation)
    job.write_model(DUB_TRACK, DubTrack(lines=lines))
    assemble_timeline(job, lines)
    return line


def _as_segment(t: TranslatedSegment) -> Segment:
    return Segment(id=t.id, start=t.start, end=t.end, speaker=t.speaker, text=t.source_text)
