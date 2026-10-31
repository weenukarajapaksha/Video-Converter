"""Gemini structured-output helper with error classification and cost tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from dubber.config import get_settings
from dubber.errors import DubberError, TransientError
from dubber.pricing import gemini_text_cost

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def cost(self, model: str) -> float:
        return gemini_text_cost(model, self.input_tokens, self.output_tokens)


def client():
    from google import genai

    key = get_settings().gemini_api_key
    if not key:
        raise DubberError("GEMINI_API_KEY is not set (see .env.example)")
    return genai.Client(api_key=key)


def classify(exc: Exception) -> Exception:
    """Map SDK / network exceptions onto retryable vs permanent pipeline errors."""
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        if exc.code in _RETRYABLE_HTTP:
            return TransientError(f"Gemini {exc.code}: {exc.message}")
        return DubberError(f"Gemini {exc.code}: {exc.message}")
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return TransientError(f"network error: {exc}")
    return exc


def generate(model: str, system: str, prompt: str, schema: type[T], usage: Usage,
             temperature: float = 0.4, thinking_budget: int = 1024) -> T:
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )
    try:
        resp = client().models.generate_content(model=model, contents=prompt, config=config)
    except Exception as exc:  # noqa: BLE001 - re-raised as a classified error
        raise classify(exc) from exc

    meta = resp.usage_metadata
    if meta is not None:
        usage.input_tokens += meta.prompt_token_count or 0
        usage.output_tokens += (meta.candidates_token_count or 0) + (
            meta.thoughts_token_count or 0)
    if resp.parsed is None:
        raise TransientError(f"model returned unparseable output: {(resp.text or '')[:300]}")
    return resp.parsed
