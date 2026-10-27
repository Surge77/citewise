from types import SimpleNamespace

import pytest
from google.genai import errors

from citewise.config import get_api_key
from citewise.llm import GeminiLLM, LLMError


class StubModels:
    def __init__(self, outcome: object):
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def gemini_with(outcome: object) -> tuple[GeminiLLM, StubModels]:
    llm = GeminiLLM(api_key="test-key", model="gemini-test")
    models = StubModels(outcome)
    llm._client = SimpleNamespace(models=models)  # type: ignore[assignment]
    return llm, models


class TestGeminiLLM:
    def test_returns_response_text_using_configured_model_at_temperature_zero(self):
        llm, models = gemini_with(SimpleNamespace(text="An answer [1]."))

        assert llm.generate("prompt") == "An answer [1]."
        assert models.calls[0]["model"] == "gemini-test"
        assert models.calls[0]["config"].temperature == 0  # type: ignore[attr-defined]

    def test_blocked_or_empty_response_raises_llm_error(self):
        llm, _ = gemini_with(SimpleNamespace(text=None))

        with pytest.raises(LLMError, match="no text"):
            llm.generate("prompt")

    def test_api_error_becomes_llm_error_with_status_only(self):
        llm, _ = gemini_with(errors.APIError(429, {"error": {"message": "quota for key AIza-secret"}}))

        with pytest.raises(LLMError) as raised:
            llm.generate("prompt")

        assert "429" in str(raised.value)
        assert "AIza" not in str(raised.value)


class TestGetApiKey:
    def test_prefers_gemini_api_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini")
        monkeypatch.setenv("GOOGLE_API_KEY", "google")

        assert get_api_key() == "gemini"

    def test_falls_back_to_google_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google")

        assert get_api_key() == "google"

    def test_raises_clear_error_when_no_key_is_set(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            get_api_key()
