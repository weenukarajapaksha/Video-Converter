"""Timing logic: speaker-aware segmentation, time budgets and fit decisions."""

import pytest

from dubber.align import Turn, assign_speakers, plan_fit, time_budgets, word_budget
from dubber.schemas import Segment, Word


def words(*spec):
    """spec: (start, end, text) tuples."""
    return [Word(start=s, end=e, text=t, probability=0.9) for s, e, t in spec]


def test_segment_spanning_two_speakers_is_split_at_speaker_change():
    seg = Segment(id=0, start=1.0, end=6.0, text="你今天为什么来这里？我想和你谈谈。", words=words(
        (1.0, 1.5, "你今天"), (1.5, 2.2, "为什么"), (2.2, 3.0, "来这里？"),
        (3.2, 4.0, "我想"), (4.0, 4.8, "和你"), (4.8, 6.0, "谈谈。"),
    ))
    turns = [Turn(0.9, 3.1, "spk_b"), Turn(3.1, 6.2, "spk_a")]

    lines = assign_speakers([seg], turns)

    assert [(ln.speaker, ln.text) for ln in lines] == [
        ("SPEAKER_01", "你今天为什么来这里？"),  # relabelled by order of appearance
        ("SPEAKER_02", "我想和你谈谈。"),
    ]
    assert (lines[0].start, lines[0].end) == (1.0, 3.0)
    assert (lines[1].start, lines[1].end) == (3.2, 6.0)


def test_short_diarization_blip_does_not_create_a_fake_line():
    seg = Segment(id=0, start=0.0, end=3.0, text="一二三四", words=words(
        (0.0, 0.7, "一"), (0.7, 0.9, "二"), (0.9, 2.0, "三"), (2.0, 3.0, "四"),
    ))
    turns = [Turn(0.0, 0.7, "A"), Turn(0.7, 0.9, "B"), Turn(0.9, 3.0, "A")]
    lines = assign_speakers([seg], turns)
    assert len(lines) == 1 and lines[0].text == "一二三四"


def test_long_pause_splits_same_speaker():
    seg = Segment(id=0, start=0, end=5, text="好的。走吧", words=words(
        (0.0, 1.0, "好的。"), (3.0, 5.0, "走吧")))
    lines = assign_speakers([seg], [Turn(0, 5, "A")])
    assert [ln.text for ln in lines] == ["好的。", "走吧"]


def test_budgets_extend_into_silence_but_not_into_next_line():
    segs = [
        Segment(id=0, start=1.0, end=2.0, text="a"),  # next line starts at 2.5
        Segment(id=1, start=2.5, end=4.0, text="b"),  # long silence after, capped at +1.5s
        Segment(id=2, start=9.0, end=9.5, text="c"),  # video ends at 9.8
    ]
    assert time_budgets(segs, total_duration=9.8) == [1.35, 3.0, 0.8]


def test_overlapping_source_lines_keep_their_own_duration():
    segs = [Segment(id=0, start=0.0, end=3.0, text="a"),
            Segment(id=1, start=2.0, end=4.0, text="b")]
    assert time_budgets(segs, total_duration=10)[0] == pytest.approx(3.0)


@pytest.mark.parametrize("duration, budget, rewrites, expected", [
    (2.0, 2.5, 0, ("ok", 1.0)),
    (2.2, 2.0, 0, ("stretch", 1.1)),  # 10% over: inaudible stretch
    (3.0, 2.0, 0, ("rewrite", 1.0)),  # 50% over: ask the LLM for a shorter line
    (3.0, 2.0, 2, ("stretch", 1.25)),  # rewrites exhausted: stretch, but capped
])
def test_plan_fit_prefers_rewrite_over_heavy_time_stretch(duration, budget, rewrites, expected):
    plan = plan_fit(duration, budget, rewrites)
    assert (plan.action, plan.tempo) == expected


def test_word_budget_scales_with_time():
    assert word_budget(1.0) == 2
    assert word_budget(4.0) == 10
    assert word_budget(0.1) == 1
