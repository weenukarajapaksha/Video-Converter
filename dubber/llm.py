"""Gemini structured-output helper: fast failover, circuit breaker, cost tracking.

Latency design: the SDK's own retry loop is disabled (it silently retried 503s with
long backoff, which once cost 6 minutes on a single translation). Instead each
request fails fast and moves down a fallback chain, and a model that just returned
a transient error is skipped for ``COOLDOWN_S`` (circuit breaker), so later calls
in the same job go straight to a healthy model.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel

from dubber.config import get_settings
from dubber.errors import DubberError, TransientError
from dubber.pricing import gemini_text_cost

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

_RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}
# Tried in order when the requested model is overloaded (503) or rate-limited (429).
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-flash-latest"]
COOLDOWN_S = 120.0
# A healthy full-script translation takes ~8-40 s; an overloaded model can hang
# without erroring, so fail over after 60 s (the circuit breaker then skips it).
REQUEST_TIMEOUT_MS = 60_000

class QuotaExhausted(DubberError):
    """Daily quota used up: try another model, but never retry the same one."""


_cooldown: dict[str, float] = {}
_cooldown_lock = threading.Lock()


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None  # model that actually answered (may be a fallback)

    def cost(self, model: str) -> float:
        return gemini_text_cost(self.model or model, self.input_tokens, self.output_tokens)


@lru_cache(maxsize=1)
def client():
    """One shared client: a temporary one is closed on garbage collection mid-request."""
    from google import genai
    from google.genai import types

    key = get_settings().gemini_api_key
    if not key:
        raise DubberError("GEMINI_API_KEY is not set (see .env.example)")
    return genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=REQUEST_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)))


def classify(exc: Exception) -> Exception:
    """Map SDK / network exceptions onto retryable vs permanent pipeline errors."""
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        if exc.code == 429 and "PerDay" in str(exc):
            # A daily quota will not come back within this run: retrying wastes minutes.
            return QuotaExhausted(f"Gemini daily quota exhausted for this model: {exc.message}")
        if exc.code in _RETRYABLE_HTTP:
            return TransientError(f"Gemini {exc.code}: {exc.message}")
        return DubberError(f"Gemini {exc.code}: {exc.message}")
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return TransientError(f"network error: {exc}")
    if type(exc).__module__.startswith("httpx"):  # ReadTimeout, ConnectError, ...
        return TransientError(f"network error: {exc!r}")
    return exc


def candidate_models(model: str, now: float | None = None) -> list[str]:
    """Requested model first, then fallbacks; models in cooldown go to the back."""
    now = time.monotonic() if now is None else now
    chain = [model, *(m for m in FALLBACK_MODELS if m != model)]
    with _cooldown_lock:
        healthy = [m for m in chain if _cooldown.get(m, 0.0) <= now]
    return healthy + [m for m in chain if m not in healthy]


def _trip(model: str) -> None:
    with _cooldown_lock:
        _cooldown[model] = time.monotonic() + COOLDOWN_S


def generate(model: str, system: str, prompt: str, schema: type[T], usage: Usage,
             temperature: float = 0.4, thinking: str | None = None) -> T:
    """``thinking``: "low" for latency-sensitive calls (Gemini 3 models only)."""
    from google.genai import types

    candidates = candidate_models(model)
    for i, name in enumerate(candidates):
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            thinking_config=(types.ThinkingConfig(thinking_level=thinking)
                             if thinking and name.startswith("gemini-3") else None),
        )
        try:
            resp = client().models.generate_content(model=name, contents=prompt, config=config)
            usage.model = name
            break
        except Exception as exc:  # noqa: BLE001 - re-raised as a classified error
            err = classify(exc)
            if isinstance(err, (TransientError, QuotaExhausted)):
                _trip(name)
                if i + 1 < len(candidates):
                    log.warning("%s unavailable (%s); falling back to %s",
                                name, err, candidates[i + 1])
                    continue
                if isinstance(err, QuotaExhausted):
                    err = QuotaExhausted(
                        "Gemini daily free-tier quota is exhausted for every fallback model. "
                        "Enable billing on the API key, set DUBBER_TRANSLATION_MODEL to a "
                        "-lite model, or retry after the daily reset.")
            raise err from exc

    meta = resp.usage_metadata
    if meta is not None:
        usage.input_tokens += meta.prompt_token_count or 0
        usage.output_tokens += (meta.candidates_token_count or 0) + (
            meta.thoughts_token_count or 0)
    if resp.parsed is None:
        raise TransientError(f"model returned unparseable output: {(resp.text or '')[:300]}")
    return resp.parsed
