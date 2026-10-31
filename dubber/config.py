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

    whisper_model: str = "large-v3"
    translation_model: str = "gemini-2.5-flash"
    tts_provider: Literal["gemini", "edge"] = "gemini"
    target_language: str = "English"


@lru_cache
def get_settings() -> Settings:
    return Settings()
