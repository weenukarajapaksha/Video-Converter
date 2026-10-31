"""SRT generation."""

from __future__ import annotations

import textwrap
from collections.abc import Iterable


def _ts(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap(text: str, width: int = 42, cjk: bool = False) -> str:
    """Max two lines; CJK has no spaces so wrap by character count."""
    text = " ".join(text.split())
    if cjk:
        width = 18
        return "\n".join(text[i:i + width] for i in range(0, len(text), width))
    lines = textwrap.wrap(text, width) or [""]
    if len(lines) > 2:  # rebalance into two lines rather than three short ones
        lines = textwrap.wrap(text, (len(text) + 1) // 2 + 1)
    return "\n".join(lines)


def to_srt(cues: Iterable[tuple[float, float, str]], cjk: bool = False) -> str:
    """cues: (start, end, text). Empty cues are skipped; overlaps are clipped."""
    ordered = sorted((c for c in cues if c[2].strip()), key=lambda c: c[0])
    blocks = []
    for n, (start, end, text) in enumerate(ordered, 1):
        if n < len(ordered):
            end = min(end, ordered[n][0] - 0.001)
        end = max(end, start + 0.3)
        blocks.append(f"{n}\n{_ts(start)} --> {_ts(end)}\n{wrap(text, cjk=cjk)}\n")
    return "\n".join(blocks)
