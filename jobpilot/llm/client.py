"""Provider-agnostic LLM wrapper.

Tools only ever call two methods:
    client.complete(prompt, system=...)            -> str
    client.structured(prompt, Schema, system=...)  -> Schema instance

Each provider subclass implements one private method, `_generate`. That is the
only provider-specific code, so swapping Gemini for OpenAI/Ollama later means
adding one small class, not touching the tools.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from jobpilot.config import ConfigError, Settings, get_settings
from jobpilot.llm import prompts

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """The LLM kept returning output that failed schema validation."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Usage:
    """Running totals, used later for the cost/speed metrics in the eval."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class LLMClient(ABC):
    def __init__(self, model: str, max_retries: int = 2) -> None:
        self.model = model
        self.max_retries = max_retries
        self.usage = Usage()

    @abstractmethod
    def _generate(self, prompt: str, *, system: str | None, json_mode: bool, temperature: float) -> LLMResponse:
        """Send one request to the provider. Subclasses implement this."""

    def _call(self, prompt: str, *, system: str | None, json_mode: bool, temperature: float) -> str:
        start = time.perf_counter()
        resp = self._generate(prompt, system=system, json_mode=json_mode, temperature=temperature)
        elapsed = time.perf_counter() - start

        self.usage.calls += 1
        self.usage.input_tokens += resp.input_tokens
        self.usage.output_tokens += resp.output_tokens
        self.usage.seconds += elapsed
        logger.debug(
            "LLM call model=%s in=%d out=%d %.2fs", self.model, resp.input_tokens, resp.output_tokens, elapsed
        )
        return resp.text

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        """Free-text answer (e.g. a score explanation or cover letter)."""
        return self._call(prompt, system=system, json_mode=False, temperature=temperature).strip()

    def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        """Return a validated `schema` instance.

        If the model returns invalid JSON or JSON that fails validation, we
        send the error back and ask it to fix its answer, up to max_retries times.
        """
        base_prompt = f"{prompt}\n\n{prompts.json_instructions(schema)}"
        current_prompt = base_prompt
        last_error: ValidationError | None = None

        for attempt in range(self.max_retries + 1):
            raw = self._call(current_prompt, system=system, json_mode=True, temperature=temperature)
            try:
                return schema.model_validate_json(_strip_code_fences(raw))
            except ValidationError as err:
                last_error = err
                logger.warning("Invalid %s output (attempt %d): %s", schema.__name__, attempt + 1, err)
                current_prompt = base_prompt + "\n\n" + prompts.RETRY_TEMPLATE.format(previous=raw, errors=err)

        raise StructuredOutputError(
            f"{schema.__name__} still invalid after {self.max_retries + 1} attempts: {last_error}"
        )


def _strip_code_fences(text: str) -> str:
    """Some models wrap JSON in ```json ... ``` even when told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str, max_retries: int = 2) -> None:
        super().__init__(model, max_retries)
        from google import genai  # imported here so tests don't need the SDK configured
        from google.genai import types

        self._types = types
        # v1beta is required for system_instruction (the v1 endpoint rejects it).
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))

    def _generate(self, prompt: str, *, system: str | None, json_mode: bool, temperature: float) -> LLMResponse:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        meta = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            input_tokens=(meta.prompt_token_count or 0) if meta else 0,
            output_tokens=(meta.candidates_token_count or 0) if meta else 0,
        )


def get_client(settings: Settings | None = None) -> LLMClient:
    """Build the client for the provider chosen in .env (LLM_PROVIDER)."""
    settings = settings or get_settings()

    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.llm_max_retries)

    raise ConfigError(f"No client for provider {settings.llm_provider!r}")
