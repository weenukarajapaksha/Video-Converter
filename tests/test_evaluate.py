"""Evaluation metrics (scripts/evaluate.py) on tiny synthetic inputs."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate as ev  # noqa: E402


@pytest.mark.parametrize("a, b, dist", [
    ("", "", 0), ("abc", "", 3), ("kitten", "sitting", 3),
    ("唐代木构", "丹代木构", 1), ("你好", "你好吗", 1),
])
def test_levenshtein(a, b, dist):
    assert ev.levenshtein(a, b) == dist == ev.levenshtein(b, a)


def test_cer_ignores_punctuation_and_whitespace():
    assert ev.cer("你今天为什么来这里？", "你今天 为什么来这里?") == 0.0
    assert ev.cer("中国已经没有唐代木构。", "中国已经没有丹代木构") == pytest.approx(1 / 10)
    assert ev.cer("", "") == 0.0 and ev.cer("", "多余") == 1.0


def test_coverage_counts_overlapping_hypotheses_once():
    truth = [(0.0, 10.0), (20.0, 30.0)]
    hyp = [(0.0, 6.0), (4.0, 10.0), (20.0, 25.0)]  # overlap between first two
    assert ev.coverage(truth, hyp) == pytest.approx(15 / 20)
    assert ev.coverage(truth, []) == 0.0


def test_silences_include_lead_in_and_tail():
    assert ev.silences([(1.5, 3.0), (3.5, 5.0), (12.0, 14.0)], duration=16.0) == [
        (0.0, 1.5), (5.0, 12.0), (14.0, 16.0)]


def test_silence_leaks_allow_designed_spill_over_but_flag_lines_in_the_gap():
    gaps = [(5.0, 12.0)]
    ok = [(4.0, 6.4)]  # starts before the gap, runs 1.4 s into it: allowed
    leak = [(8.0, 9.0)]  # starts in the protected middle of the silence
    assert ev.silence_leaks(gaps, ok) == []
    assert ev.silence_leaks(gaps, ok + leak) == leak


def test_rms_dbfs():
    assert ev.rms_dbfs(np.zeros(100)) == float("-inf")
    assert ev.rms_dbfs(np.full(100, 0.5)) == pytest.approx(-6.02, abs=0.01)
