from typing import Protocol

from google import genai
from google.genai import errors, types


class LLMError(Exception):
    """The language model could not produce an answer."""


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class GeminiLLM:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
        except errors.APIError as exc:
            raise LLMError(f"Gemini request failed with status {exc.code}") from exc
        if not response.text:
            raise LLMError("Gemini returned no text; the answer may have been blocked by safety filters")
        return response.text
