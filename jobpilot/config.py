"""Settings loaded from environment variables (and the .env file).

All config lives here so no other module reads os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# Providers with a working client. openai / anthropic / ollama come later.
SUPPORTED_PROVIDERS = ("gemini",)


class ConfigError(Exception):
    """Raised when required settings are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    gemini_api_key: str | None
    gemini_model: str
    llm_max_retries: int


@lru_cache
def get_settings() -> Settings:
    """Read settings once and cache them for the rest of the run."""
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER={provider!r} is not supported yet. Choose one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    return Settings(
        llm_provider=provider,
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )
