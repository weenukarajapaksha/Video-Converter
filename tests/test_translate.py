"""Translation logic with the LLM stubbed out (no network)."""

import pytest

from dubber import llm, translate
from dubber.schemas import Segment
from dubber.translate import IssueOut, LineOut, LinesOut, ReviewOut


def segs():
    return [Segment(id=0, start=0, end=2, speaker="SPEAKER_01", text="你今天为什么来这里？"),
            Segment(id=1, start=2.5, end=4, speaker="SPEAKER_02", text="解铃还须系铃人。")]


def test_missing_lines_are_requested_again(monkeypatch):
    replies = iter([
        LinesOut(lines=[LineOut(id=0, text="Why are you here today?", emotion="Serious")]),
        LinesOut(lines=[LineOut(id=1, text="I'll fix what I broke.", emotion="whatever")]),
    ])
    prompts = []

    def fake_generate(model, system, prompt, schema, usage, **kw):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(llm, "generate", fake_generate)
    out = translate.translate_lines(segs(), [2.0, 2.0], "m", "English", llm.Usage())

    assert out[0].text == "Why are you here today?"
    assert out[0].emotion == "serious"  # normalised
    assert out[1].emotion == "neutral"  # unknown emotion falls back
    assert "skipped ids [1]" in prompts[1]


def test_critic_fixes_that_break_the_time_budget_are_discarded(monkeypatch):
    draft = {0: LineOut(id=0, text="Why are you here today?", emotion="neutral"),
             1: LineOut(id=1, text="The bell must be untied by whoever tied it.", emotion="calm")}
    review = ReviewOut(accuracy=7, naturalness=5, tone=8, summary="literal idiom", issues=[
        IssueOut(id=1, category="idiom", comment="literal", revised_text="I'll fix this."),
        IssueOut(id=0, category="fluency", comment="x",
                 revised_text="So tell me, what exactly brings you all the way here today?"),
        IssueOut(id=9, category="accuracy", comment="hallucinated id", revised_text="Hi."),
    ])
    monkeypatch.setattr(llm, "generate", lambda *a, **k: review)

    out = translate.critique(segs(), [2.0, 2.0], draft, "m", "English", llm.Usage())

    assert [i.id for i in out.issues] == [1]  # over-budget fix and unknown id dropped
    assert out.issues[0].revised_text == "I'll fix this."


def test_usage_cost_uses_the_model_that_actually_answered():
    usage = llm.Usage(input_tokens=1_000_000, output_tokens=0, model="gemini-2.5-flash")
    assert usage.cost("gemini-3.8-flash") == pytest.approx(0.30)
