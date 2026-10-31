"""Runtime settings, loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DUBBER_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Secrets use their conventional, unprefixed names.
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")

    jobs_dir: Path = Path("jobs")
    device: Literal["auto", "cuda", "cpu"] = "auto"

    # Input constraints (Task 1).
    min_duration_s: float = 30.0
    max_duration_s: float = 180.0

    separation_model: str = "htdemucs_ft"
    whisper_model: str = "large-v3"
    # Batched decoding is ~3x faster on long videos but decodes chunks without
    # cross-chunk context; sequential + coverage recovery is the accuracy default.
    asr_batched: bool = False
    translation_model: str = "gemini-3.8-flash"
    tts_model: str = "gemini-3.8-flash-tts"
    # Edge is free and unlimited; Gemini TTS is more expressive but its free tier
    # allows very few requests per day.
    tts_provider: Literal["gemini", "edge"] = "edge"
    target_language: str = "English"


@lru_cache
def get_settings() -> Settings:
    return Settings()
