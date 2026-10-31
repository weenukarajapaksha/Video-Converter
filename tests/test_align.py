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


def test_speaker_change_one_syllable_off_snaps_to_punctuation():
    # Real case: diarization switched speaker one syllable late -> "回去？回 | 中国".
    seg = Segment(id=0, start=19.9, end=23.0, text="回去？回中国，", words=words(
        (19.9, 20.4, "回去？"), (20.6, 20.9, "回"), (21.1, 21.6, "中国，")))
    turns = [Turn(19.8, 20.95, "B"), Turn(20.95, 23.0, "A")]
    lines = assign_speakers([seg], turns)
    assert [ln.text for ln in lines] == ["回去？", "回中国，"]


def test_stray_fragment_is_merged_into_following_line_of_same_speaker():
    # Real case: a bad word timestamp left "日本" 2 s before "学者断言…".
    segs = [Segment(id=0, start=7.0, end=7.4, text="日本", words=words((7.0, 7.4, "日本"))),
            Segment(id=1, start=9.6, end=12.0, text="学者断言。", words=words(
                (9.6, 10.5, "学者"), (10.5, 12.0, "断言。")))]
    lines = assign_speakers(segs, [Turn(6.9, 12.1, "A")])
    assert [(ln.start, ln.text) for ln in lines] == [(7.0, "日本学者断言。")]


def test_micro_lines_merge_into_same_speaker_neighbour_but_not_across_speakers():
    # Real lines from the film clip (jobs/debug-run/transcript.json): "没有测绘" (0.44 s)
    # was dubbed "Blank." and "别" (0.58 s) "Don't." because their slots were tiny.
    from dubber.align import merge_short_lines

    lines = [
        Segment(id=0, start=34.64, end=35.08, speaker="S4", text="没有测绘"),
        Segment(id=1, start=35.08, end=37.32, speaker="S4", text="建筑学完全是一片空白"),
        Segment(id=2, start=37.32, end=37.90, speaker="S4", text="别"),
        Segment(id=3, start=37.90, end=40.12, speaker="S5", text="找了几千年了"),
    ]
    merged = merge_short_lines(lines)
    assert [(s.speaker, s.start, s.end, s.text) for s in merged] == [
        ("S4", 34.64, 37.90, "没有测绘建筑学完全是一片空白别"),
        ("S5", 37.90, 40.12, "找了几千年了"),  # different speaker: real turn, kept
    ]
    assert [s.id for s in merged] == [0, 1]


def test_unpunctuated_fragment_joins_the_sentence_it_starts():
    # Real case: "日本人的话" sits between two lines with 0 s gaps; it begins
    # "日本人的话虽然难听", so it must join the next line, not the previous one.
    from dubber.align import merge_short_lines

    lines = [Segment(id=0, start=43.78, end=45.72, speaker="S5", text="唐构建筑早就没了"),
             Segment(id=1, start=45.72, end=46.70, speaker="S5", text="日本人的话"),
             Segment(id=2, start=46.70, end=47.98, speaker="S5", text="虽然难听")]
    assert [s.text for s in merge_short_lines(lines)] == ["唐构建筑早就没了", "日本人的话虽然难听"]


def test_short_line_isolated_by_a_long_pause_is_kept():
    from dubber.align import merge_short_lines

    lines = [Segment(id=0, start=0.0, end=0.5, speaker="A", text="好。"),
             Segment(id=1, start=2.0, end=4.0, speaker="A", text="我们走吧。")]
    assert [s.text for s in merge_short_lines(lines)] == ["好。", "我们走吧。"]


def test_short_line_gets_a_minimum_budget_from_the_gap_before_the_next_line():
    # 0.4 s line, next line 0.6 s later: the 0.15 s breathing pause would leave 0.45 s;
    # the floor gives it 0.8 s (still ends before the next line starts).
    segs = [Segment(id=0, start=10.0, end=10.4, text="a"),
            Segment(id=1, start=10.9, end=12.0, text="b")]
    assert time_budgets(segs, total_duration=20)[0] == pytest.approx(0.8)
    # ...but never runs into the next line when the gap is smaller than the floor.
    segs[1] = segs[1].model_copy(update={"start": 10.6})
    assert time_budgets(segs, total_duration=20)[0] == pytest.approx(0.6)


def test_uncovered_speech_finds_regions_asr_skipped():
    from dubber.align import uncovered_speech

    segs = [Segment(id=0, start=0.0, end=6.8, text="a"), Segment(id=1, start=34.6, end=50, text="b")]
    speech = [(0.0, 7.0), (9.5, 34.0), (34.5, 50.0), (51.0, 51.3)]
    assert uncovered_speech(speech, segs) == [(9.5, 34.0)]  # short blip ignored


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
