from collections.abc import Iterator
from typing import Protocol

from google import genai
from google.genai import errors, types


class LLMError(Exception):
    """The language model could not produce an answer."""


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...
    def stream(self, prompt: str) -> Iterator[str]: ...


NO_TEXT_MESSAGE = "Gemini returned no text; the answer may have been blocked by safety filters"


class GeminiLLM:
    def __init__(self, api_key: str, model: str, thinking_budget: int | None = None) -> None:
        """thinking_budget=None keeps the model's default; 0 turns Gemini 2.5 Flash's hidden reasoning off."""
        self._client = genai.Client(api_key=api_key)
        self._model = model
        thinking = None if thinking_budget is None else types.ThinkingConfig(thinking_budget=thinking_budget)
        self._config = types.GenerateContentConfig(temperature=0, thinking_config=thinking)

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(model=self._model, contents=prompt, config=self._config)
        except errors.APIError as exc:
            raise LLMError(f"Gemini request failed with status {exc.code}") from exc
        if not response.text:
            raise LLMError(NO_TEXT_MESSAGE)
        return response.text

    def stream(self, prompt: str) -> Iterator[str]:
        produced_text = False
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self._model, contents=prompt, config=self._config
            ):
                if chunk.text:
                    produced_text = True
                    yield chunk.text
        except errors.APIError as exc:
            raise LLMError(f"Gemini request failed with status {exc.code}") from exc
        if not produced_text:
            raise LLMError(NO_TEXT_MESSAGE)
