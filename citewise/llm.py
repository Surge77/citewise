import logging
from collections.abc import Iterator
from typing import Protocol

from google import genai
from google.genai import errors, types

logger = logging.getLogger(__name__)

RATE_LIMITED_STATUS = 429
NO_TEXT_MESSAGE = "Gemini returned no text; the answer may have been blocked by safety filters"


class LLMError(Exception):
    """The language model could not produce an answer."""


class LLMRateLimitedError(LLMError):
    """Every configured model is out of quota."""


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...
    def stream(self, prompt: str) -> Iterator[str]: ...


def _config_for(model: str, thinking_budget: int | None) -> types.GenerateContentConfig:
    # Only Gemini 2.5 models accept thinking_budget; Gemini 3.x rejects it with HTTP 400 (checked against the API).
    use_budget = thinking_budget is not None and model.startswith("gemini-2.5")
    thinking = types.ThinkingConfig(thinking_budget=thinking_budget) if use_budget else None
    return types.GenerateContentConfig(temperature=0, thinking_config=thinking)


def _api_failure(exc: errors.APIError) -> LLMError:
    # Status only: Gemini's messages can include quota details and key fragments that must not reach users.
    return LLMError(f"Gemini request failed with status {exc.code}")


class GeminiLLM:
    """Gemini client that falls back to the next model when one is out of quota (each model has its own quota)."""

    def __init__(self, api_key: str, models: list[str], thinking_budget: int | None = None) -> None:
        if not models:
            raise ValueError("GeminiLLM needs at least one model")
        self._client = genai.Client(api_key=api_key)
        self._models = models
        self._configs = {model: _config_for(model, thinking_budget) for model in models}

    def generate(self, prompt: str) -> str:
        for model in self._models:
            try:
                response = self._client.models.generate_content(
                    model=model, contents=prompt, config=self._configs[model]
                )
            except errors.APIError as exc:
                if exc.code == RATE_LIMITED_STATUS:
                    logger.warning("%s is rate limited; trying the next model", model)
                    continue
                raise _api_failure(exc) from exc
            if not response.text:
                raise LLMError(NO_TEXT_MESSAGE)
            return response.text
        raise LLMRateLimitedError("every configured Gemini model is rate limited")

    def stream(self, prompt: str) -> Iterator[str]:
        for model in self._models:
            produced_text = False
            try:
                for chunk in self._client.models.generate_content_stream(
                    model=model, contents=prompt, config=self._configs[model]
                ):
                    if chunk.text:
                        produced_text = True
                        yield chunk.text
            except errors.APIError as exc:
                # Switching models is only safe before any text has been sent to the user.
                if exc.code == RATE_LIMITED_STATUS and not produced_text:
                    logger.warning("%s is rate limited; trying the next model", model)
                    continue
                raise _api_failure(exc) from exc
            if not produced_text:
                raise LLMError(NO_TEXT_MESSAGE)
            return
        raise LLMRateLimitedError("every configured Gemini model is rate limited")
