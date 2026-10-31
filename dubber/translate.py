"""LLM dubbing-script adaptation (Gemini, structured JSON output).

Translation for dubbing is not plain MT: every line has a hard time budget, so the
model gets the whole conversation for context plus, per line, the speaker, the
seconds available and a word budget, and returns natural spoken English plus an
emotion tag that drives expressive TTS.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from dubber import llm
from dubber.align import word_budget
from dubber.schemas import Segment

EMOTIONS = ["neutral", "happy", "excited", "sad", "angry", "surprised", "fearful",
            "calm", "serious", "tender", "sarcastic", "worried"]

CHUNK_LINES = 60  # lines per request; long videos are processed in windows
CONTEXT_LINES = 8  # previously translated lines shown for continuity


class LineOut(BaseModel):
    id: int
    text: str = Field(description="the spoken target-language line")
    emotion: str = Field(description=f"one of: {', '.join(EMOTIONS)}")


class LinesOut(BaseModel):
    lines: list[LineOut]


SYSTEM = """You are a senior dubbing script adapter localizing Chinese film/video dialogue into {lang}.
Write lines that actors would really say, to be voiced by TTS and lip-synced to the original timing.

Rules:
- Preserve meaning, intent, subtext and emotional tone. Keep the speaker's register (formal, casual, rude, playful).
- Translate idioms, chengyu and slang into natural {lang} equivalents, never word-for-word.
- Use the whole conversation for context: resolve pronouns, keep names and terms consistent, and make replies follow from the previous lines.
- HARD LIMIT: each line must not exceed its max_words. If the meaning cannot fit, simplify or drop filler; never cut key information.
- Prefer contractions and short spoken phrasing. No stage directions, notes, quotes or brackets.
- Keep one output line per input id; never merge or split lines.
- emotion: the delivery emotion of the ORIGINAL line, one of: {emotions}."""


def _line_spec(seg: Segment, budget: float) -> str:
    return (f'[{seg.id}] {seg.speaker} {seg.start:.1f}-{seg.end:.1f}s '
            f'(max {budget:.1f}s, max_words={word_budget(budget)}): {seg.text}')


def translate_lines(segments: list[Segment], budgets: list[float], model: str,
                    target_language: str, usage: llm.Usage,
                    on_chunk=lambda done, total: None) -> dict[int, LineOut]:
    system = SYSTEM.format(lang=target_language, emotions=", ".join(EMOTIONS))
    result: dict[int, LineOut] = {}
    for lo in range(0, len(segments), CHUNK_LINES):
        chunk = list(zip(segments[lo:lo + CHUNK_LINES], budgets[lo:lo + CHUNK_LINES]))
        context = segments[max(0, lo - CONTEXT_LINES):lo]
        upcoming = segments[lo + CHUNK_LINES:lo + CHUNK_LINES + CONTEXT_LINES]

        parts = []
        if context:
            parts.append("Previous lines (already translated, for context only):\n" + "\n".join(
                f"[{s.id}] {s.speaker}: {s.text} => {result[s.id].text}" for s in context))
        parts.append("Translate these lines:\n" + "\n".join(_line_spec(s, b) for s, b in chunk))
        if upcoming:
            parts.append("Following lines (context only, do not translate):\n" + "\n".join(
                f"{s.speaker}: {s.text}" for s in upcoming))

        out = _request(model, system, "\n\n".join(parts), {s.id for s, _ in chunk}, usage)
        result.update(out)
        on_chunk(min(lo + CHUNK_LINES, len(segments)), len(segments))
    return result


class IssueOut(BaseModel):
    id: int
    category: str = Field(description="accuracy | idiom | tone | fluency | consistency | length")
    comment: str = Field(description="what is wrong, in one sentence")
    revised_text: str = Field(description="the corrected line, within its max_words")


class ReviewOut(BaseModel):
    accuracy: int = Field(description="1-10: meaning preserved")
    naturalness: int = Field(description="1-10: sounds like native spoken dialogue")
    tone: int = Field(description="1-10: emotion and register preserved")
    summary: str = Field(description="one or two sentences")
    issues: list[IssueOut] = Field(description="only lines that genuinely need fixing")


CRITIC_SYSTEM = """You are a strict dubbing QA reviewer checking a Chinese -> {lang} dub script.
Compare every translated line with its source and the surrounding conversation. Flag only real problems:
mistranslations or lost meaning, idioms translated literally, wrong emotion or register, unnatural or
written-style phrasing, inconsistent names/terms, or lines over their max_words budget.
For each flagged line give a corrected version that fixes the problem and stays within max_words.
Do not flag lines that are already good; stylistic preferences are not issues.
Score the script as a whole (after your fixes are NOT applied) on 1-10 scales."""


def critique(segments: list[Segment], budgets: list[float], lines: dict[int, LineOut],
             model: str, target_language: str, usage: llm.Usage) -> ReviewOut:
    """LLM-as-judge pass: scores the draft and proposes fixes for flagged lines."""
    body = "\n".join(
        f"[{s.id}] {s.speaker} (max_words={word_budget(b)}, emotion={lines[s.id].emotion})\n"
        f"    source: {s.text}\n    draft:  {lines[s.id].text}"
        for s, b in zip(segments, budgets))
    review = llm.generate(model, CRITIC_SYSTEM.format(lang=target_language),
                          "Review this dub script:\n\n" + body, ReviewOut, usage, temperature=0.2)
    budget_by_id = {s.id: word_budget(b) for s, b in zip(segments, budgets)}
    # Keep only fixes for real lines that respect the word budget (small tolerance).
    review.issues = [i for i in review.issues if i.id in lines and i.revised_text.strip()
                     and len(i.revised_text.split()) <= budget_by_id[i.id] + 1]
    return review


def shorten_lines(items: list[tuple[Segment, str, float]], model: str, target_language: str,
                  usage: llm.Usage) -> dict[int, str]:
    """Rephrase over-long lines to fit their budgets (one request for all of them)."""
    system = SYSTEM.format(lang=target_language, emotions=", ".join(EMOTIONS))
    body = "\n".join(
        f"[{seg.id}] {seg.speaker} source: {seg.text}\n    current ({len(text.split())} words, "
        f"too long): {text}\n    rewrite to at most {word_budget(budget)} words" for
        seg, text, budget in items)
    prompt = ("These dubbed lines take too long to speak in their time slot. Rewrite each to be "
              "shorter while keeping meaning and tone; keep the same emotion.\n\n" + body)
    out = _request(model, system, prompt, {seg.id for seg, _, _ in items}, usage)
    return {i: line.text for i, line in out.items()}


def _request(model: str, system: str, prompt: str, ids: set[int],
             usage: llm.Usage) -> dict[int, LineOut]:
    got: dict[int, LineOut] = {}
    for _ in range(2):  # second pass only asks for ids the model skipped
        missing = ids - got.keys()
        if not missing:
            break
        suffix = "" if not got else (f"\n\nYou skipped ids {sorted(missing)}. "
                                     "Return ONLY those ids.")
        # Low thinking for draft/shorten calls (latency); the critic pass reasons fully.
        parsed = llm.generate(model, system, prompt + suffix, LinesOut, usage, thinking="low")
        for line in parsed.lines:
            if line.id in missing and line.text.strip():
                emotion = line.emotion.lower().strip()
                got[line.id] = LineOut(id=line.id, text=line.text.strip(),
                                       emotion=emotion if emotion in EMOTIONS else "neutral")
    if missing := ids - got.keys():
        raise llm.TransientError(f"model did not translate lines {sorted(missing)}")
    return got
