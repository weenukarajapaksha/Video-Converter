"""List prices (USD) used for per-job cost accounting.

Costs are always computed at *paid-tier* rates, even when a free-tier key is used,
so the report reflects what production processing would cost.
"""

# USD per 1M tokens: (input, output). Output includes "thinking" tokens.
GEMINI_TEXT = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
}
# TTS: input text tokens, output audio tokens (Gemini emits 25 audio tokens / second).
GEMINI_TTS = {
    "gemini-2.5-flash-preview-tts": (0.50, 10.00),
    "gemini-2.5-pro-preview-tts": (1.00, 20.00),
}
AUDIO_TOKENS_PER_SECOND = 25


def gemini_text_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rate_in, rate_out = GEMINI_TEXT.get(model, GEMINI_TEXT["gemini-2.5-flash"])
    return (input_tokens * rate_in + output_tokens * rate_out) / 1e6


def gemini_tts_cost(model: str, input_tokens: int, audio_seconds: float) -> float:
    rate_in, rate_out = GEMINI_TTS.get(model, GEMINI_TTS["gemini-2.5-flash-preview-tts"])
    return (input_tokens * rate_in + audio_seconds * AUDIO_TOKENS_PER_SECOND * rate_out) / 1e6
