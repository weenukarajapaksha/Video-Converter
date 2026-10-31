"""Pure timing logic (no ML, fully unit-tested).

* ``assign_speakers`` - merge ASR words with diarization turns and re-cut the
  transcript into single-speaker, sentence-sized dubbing lines.
* ``time_budgets``   - how long each English line may last without colliding with
  the next line.
* ``plan_fit``       - decide how to make a synthesized line fit its budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dubber.schemas import Segment, Word

SENTENCE_END = re.compile(r"[。！？!?…；;]$")
CLAUSE_END = re.compile(r"[，,、：:]$")


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def speaker_at(start: float, end: float, turns: list[Turn], tolerance: float = 0.5) -> str | None:
    """Speaker with the largest overlap with [start, end]; nearest turn as fallback."""
    best, best_ov = None, 0.0
    for t in turns:
        ov = _overlap(start, end, t.start, t.end)
        if ov > best_ov:
            best, best_ov = t.speaker, ov
    if best is not None:
        return best
    mid = (start + end) / 2
    nearest = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end)), default=None)
    if nearest and min(abs(mid - nearest.start), abs(mid - nearest.end)) <= tolerance:
        return nearest.speaker
    return None


def _smooth(labels: list[str | None], words: list[Word], min_run_s: float = 0.4) -> list[str]:
    """Fill unknown labels and absorb tiny speaker 'blips' into their neighbours."""
    out = list(labels)
    for i, lab in enumerate(out):  # forward-fill, then back-fill
        if lab is None and i > 0:
            out[i] = out[i - 1]
    for i in range(len(out) - 2, -1, -1):
        if out[i] is None:
            out[i] = out[i + 1]
    out = [lab or "SPEAKER_00" for lab in out]

    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1] == out[i]:
            j += 1
        run_s = words[j].end - words[i].start
        prev_lab = out[i - 1] if i > 0 else None
        next_lab = out[j + 1] if j + 1 < len(out) else None
        if run_s < min_run_s and (j - i) < 2 and (prev_lab or next_lab):
            fill = prev_lab if prev_lab == next_lab or next_lab is None else (prev_lab or next_lab)
            for k in range(i, j + 1):
                out[k] = fill
        i = j + 1
    return out


def _snap_to_punctuation(labels: list[str], words: list[Word], reach: int = 2) -> list[str]:
    """Move each speaker change to a nearby punctuation mark.

    Diarization boundaries are often a syllable off ("回去？回 | 中国"). People change
    turns at clause/sentence ends, so a change within ``reach`` words of punctuation
    is snapped to it ("回去？| 回中国").
    """
    out = list(labels)
    for i in range(1, len(out)):
        if out[i] == out[i - 1]:
            continue
        ends = [j for j in range(max(1, i - reach), min(len(words), i + reach + 1))
                if SENTENCE_END.search(words[j - 1].text.strip())
                or CLAUSE_END.search(words[j - 1].text.strip())]
        if not ends:
            continue
        j = min(ends, key=lambda k: abs(k - i))
        if j > i:  # previous speaker keeps talking up to the punctuation
            out[i:j] = [out[i - 1]] * (j - i)
        elif j < i:  # new speaker starts right after the punctuation
            out[j:i] = [out[i]] * (i - j)
    return out


def _merge_fragments(lines: list[Segment], max_chars: int = 2,
                     max_gap: float = 3.0) -> list[Segment]:
    """Fold tiny fragments (e.g. a stray "日本" with a bad word timestamp) into the
    following line of the same speaker, so they are translated in context."""
    out: list[Segment] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        core = re.sub(r"[\s，。？！、,.?!]", "", cur.text)
        if (nxt and len(core) <= max_chars and not SENTENCE_END.search(cur.text)
                and nxt.speaker == cur.speaker and nxt.start - cur.end <= max_gap):
            lines[i + 1] = nxt.model_copy(update={
                "start": cur.start, "text": cur.text + nxt.text, "words": cur.words + nxt.words})
        else:
            out.append(cur)
        i += 1
    return [s.model_copy(update={"id": n}) for n, s in enumerate(out)]


def assign_speakers(segments: list[Segment], turns: list[Turn], max_gap: float = 0.8,
                    max_len: float = 10.0) -> list[Segment]:
    """Split ASR segments at speaker changes, long pauses and (if long) sentence ends."""
    lines: list[Segment] = []

    def emit(words: list[Word], speaker: str, fallback: Segment) -> None:
        if not words:
            return
        text = "".join(w.text for w in words).strip()
        if not text:
            return
        probs = [w.probability for w in words if w.probability is not None]
        lines.append(Segment(
            id=len(lines), start=words[0].start, end=words[-1].end, speaker=speaker,
            text=text, words=words,
            confidence=round(sum(probs) / len(probs), 3) if probs else fallback.confidence,
        ))

    for seg in segments:
        if not seg.words:  # no word timings: label the whole segment
            label = speaker_at(seg.start, seg.end, turns) or "SPEAKER_00"
            lines.append(seg.model_copy(update={"id": len(lines), "speaker": label}))
            continue

        labels = _smooth([speaker_at(w.start, w.end, turns) for w in seg.words], seg.words)
        labels = _snap_to_punctuation(labels, seg.words)
        cur: list[Word] = [seg.words[0]]
        cur_lab = labels[0]
        for word, lab in zip(seg.words[1:], labels[1:]):
            prev = cur[-1]
            long_enough = prev.end - cur[0].start >= max_len
            split = (
                lab != cur_lab
                or word.start - prev.end > max_gap
                or (long_enough and SENTENCE_END.search(prev.text.strip()))
                or (prev.end - cur[0].start >= max_len * 1.5
                    and CLAUSE_END.search(prev.text.strip()))
            )
            if split:
                emit(cur, cur_lab, seg)
                cur, cur_lab = [], lab
            cur.append(word)
        emit(cur, cur_lab, seg)

    return relabel_speakers(_merge_fragments(lines))


def relabel_speakers(lines: list[Segment]) -> list[Segment]:
    """Rename speakers SPEAKER_01, SPEAKER_02... in order of first appearance."""
    mapping: dict[str, str] = {}
    for line in sorted(lines, key=lambda s: s.start):
        mapping.setdefault(line.speaker, f"SPEAKER_{len(mapping) + 1:02d}")
    return [s.model_copy(update={"speaker": mapping[s.speaker]}) for s in lines]


def uncovered_speech(speech: list[tuple[float, float]], segments: list[Segment],
                     min_len: float = 1.0, max_coverage: float = 0.5) -> list[tuple[float, float]]:
    """VAD speech regions the transcript (mostly) failed to cover.

    Whisper can silently skip part of a 30 s decoding window. Comparing what the VAD
    heard with what ASR produced finds those holes so they can be re-transcribed.
    """
    holes = []
    for start, end in speech:
        if end - start < min_len:
            continue
        covered = sum(_overlap(start, end, s.start, s.end) for s in segments)
        if covered / (end - start) < max_coverage:
            holes.append((start, end))
    return holes


# --- timing budgets ------------------------------------------------------------

# Natural conversational English is ~150-170 words per minute.
ENGLISH_WPS = 2.6


def time_budgets(segments: list[Segment], total_duration: float, min_pause: float = 0.15,
                 max_extend: float = 1.5) -> list[float]:
    """Max seconds each line may last: its own slot plus following silence, capped.

    A line may run past its source end into the pause before the *next* line
    (keeping ``min_pause`` of breathing room), but never more than ``max_extend``
    seconds past where the original speaker stopped. Overlapping source lines keep
    their own duration (the overlap is real, not a sync error).
    """
    ordered = sorted(range(len(segments)), key=lambda i: segments[i].start)
    budgets = [0.0] * len(segments)
    for rank, i in enumerate(ordered):
        seg = segments[i]
        next_start = (segments[ordered[rank + 1]].start - min_pause if rank + 1 < len(ordered)
                      else total_duration)
        limit = min(next_start, seg.end + max_extend, total_duration)
        budgets[i] = round(max(seg.duration, limit - seg.start), 3)
    return budgets


def word_budget(seconds: float) -> int:
    return max(1, int(seconds * ENGLISH_WPS))


# --- fitting synthesized audio -----------------------------------------------------

@dataclass(frozen=True)
class FitPlan:
    action: str  # "ok" | "stretch" | "rewrite"
    tempo: float = 1.0


def plan_fit(duration: float, budget: float, rewrites_done: int, *, max_tempo: float = 1.25,
             rewrite_over: float = 1.12, max_rewrites: int = 2) -> FitPlan:
    """Choose the least audible fix for a line that is too long.

    Small overruns are time-stretched (pitch preserved). Large overruns are first
    sent back to the LLM for a shorter phrasing, because speeding speech up by
    more than ~25% sounds unnatural. If rewrites are exhausted, stretch up to the cap.
    """
    if duration <= budget:
        return FitPlan("ok")
    ratio = duration / budget
    if ratio > rewrite_over and rewrites_done < max_rewrites:
        return FitPlan("rewrite")
    return FitPlan("stretch", round(min(ratio, max_tempo), 4))
