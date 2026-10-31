"""List prices (USD) used for per-job cost accounting.

Costs are always computed at *paid-tier* rates, even when a free-tier key is used,
so the report reflects what production processing would cost. Rates as of
2026-10 (third-party listings; where sources disagreed the higher rate is used).
"""

# USD per 1M tokens: (input, output). Output includes "thinking" tokens.
GEMINI_TEXT = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (0.50, 3.00),
    "gemini-flash-latest": (0.75, 3.75),
    "gemini-2.5-flash": (0.30, 2.50),
}
DEFAULT_TEXT = (0.75, 3.75)

# TTS: input text tokens, output audio tokens (Gemini emits 25 audio tokens / second).
GEMINI_TTS = {
    "gemini-3.8-flash-tts": (0.50, 9.00),
    "gemini-3.8-flash-lite-tts": (0.25, 4.50),
    "gemini-2.5-flash-preview-tts": (0.50, 10.00),
}
DEFAULT_TTS = (0.50, 10.00)
AUDIO_TOKENS_PER_SECOND = 25


def gemini_text_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rate_in, rate_out = GEMINI_TEXT.get(model, DEFAULT_TEXT)
    return (input_tokens * rate_in + output_tokens * rate_out) / 1e6


def gemini_tts_cost(model: str, input_tokens: int, audio_seconds: float) -> float:
    rate_in, rate_out = GEMINI_TTS.get(model, DEFAULT_TTS)
    return (input_tokens * rate_in + audio_seconds * AUDIO_TOKENS_PER_SECOND * rate_out) / 1e6
