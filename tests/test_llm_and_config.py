from types import SimpleNamespace

import pytest
from google.genai import errors

from citewise.config import DEFAULT_LLM_MODELS, get_api_key, load_settings
from citewise.llm import GeminiLLM, LLMError, LLMRateLimitedError


def api_error(status: int) -> errors.APIError:
    return errors.APIError(status, {"error": {"message": f"status {status} for key AIza-secret"}})


class StubModels:
    """Per-model outcomes: a response object, a list of stream chunks, or an exception to raise."""

    def __init__(self, outcomes: dict[str, object]):
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def _outcome(self, kwargs: dict[str, object]) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes[str(kwargs["model"])]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def generate_content(self, **kwargs: object) -> object:
        return self._outcome(kwargs)

    def generate_content_stream(self, **kwargs: object) -> object:
        chunks = self._outcome(kwargs)
        return iter(chunks)  # type: ignore[call-overload]


def gemini_with(outcomes: dict[str, object], thinking_budget: int | None = None) -> tuple[GeminiLLM, StubModels]:
    llm = GeminiLLM(api_key="test-key", models=list(outcomes), thinking_budget=thinking_budget)
    models = StubModels(outcomes)
    llm._client = SimpleNamespace(models=models)  # type: ignore[assignment]
    return llm, models


def text(value: str | None) -> SimpleNamespace:
    return SimpleNamespace(text=value)


class TestGenerate:
    def test_returns_text_from_the_first_model_at_temperature_zero(self):
        llm, models = gemini_with({"gemini-a": text("An answer [1].")})

        assert llm.generate("prompt") == "An answer [1]."
        assert models.calls[0]["model"] == "gemini-a"
        assert models.calls[0]["config"].temperature == 0  # type: ignore[attr-defined]

    def test_falls_back_to_the_next_model_when_one_is_rate_limited(self):
        llm, models = gemini_with({"gemini-a": api_error(429), "gemini-b": text("From b.")})

        assert llm.generate("prompt") == "From b."
        assert [c["model"] for c in models.calls] == ["gemini-a", "gemini-b"]

    def test_raises_rate_limited_when_every_model_is_out_of_quota(self):
        llm, _ = gemini_with({"gemini-a": api_error(429), "gemini-b": api_error(429)})

        with pytest.raises(LLMRateLimitedError):
            llm.generate("prompt")

    def test_other_api_errors_do_not_fall_back_and_hide_details(self):
        llm, models = gemini_with({"gemini-a": api_error(500), "gemini-b": text("never used")})

        with pytest.raises(LLMError) as raised:
            llm.generate("prompt")

        assert not isinstance(raised.value, LLMRateLimitedError)
        assert "500" in str(raised.value)
        assert "AIza" not in str(raised.value)
        assert len(models.calls) == 1

    def test_blocked_or_empty_response_raises_llm_error(self):
        llm, _ = gemini_with({"gemini-a": text(None)})

        with pytest.raises(LLMError, match="no text"):
            llm.generate("prompt")


class TestStream:
    def test_yields_chunk_texts_and_skips_empty_chunks(self):
        llm, _ = gemini_with({"gemini-a": [text("Hel"), text(None), text("lo [1].")]})

        assert list(llm.stream("p")) == ["Hel", "lo [1]."]

    def test_falls_back_before_any_text_is_sent(self):
        llm, models = gemini_with({"gemini-a": api_error(429), "gemini-b": [text("From b.")]})

        assert list(llm.stream("p")) == ["From b."]
        assert [c["model"] for c in models.calls] == ["gemini-a", "gemini-b"]

    def test_raises_rate_limited_when_every_model_is_out_of_quota(self):
        llm, _ = gemini_with({"gemini-a": api_error(429), "gemini-b": api_error(429)})

        with pytest.raises(LLMRateLimitedError):
            list(llm.stream("p"))

    def test_other_api_errors_become_llm_error(self):
        llm, _ = gemini_with({"gemini-a": api_error(503)})

        with pytest.raises(LLMError, match="503"):
            list(llm.stream("p"))

    def test_stream_with_no_text_at_all_raises_llm_error(self):
        llm, _ = gemini_with({"gemini-a": [text(None)]})

        with pytest.raises(LLMError, match="no text"):
            list(llm.stream("p"))


class TestThinkingConfig:
    def test_thinking_budget_is_sent_only_to_gemini_2_5_models(self):
        # Gemini 3.x rejects thinking_budget with HTTP 400 (checked against the live API).
        llm, models = gemini_with({"gemini-2.5-flash": api_error(429), "gemini-3.5-flash-lite": text("ok")}, 0)

        llm.generate("p")

        budget_2_5 = models.calls[0]["config"].thinking_config  # type: ignore[attr-defined]
        assert budget_2_5 is not None and budget_2_5.thinking_budget == 0
        assert models.calls[1]["config"].thinking_config is None  # type: ignore[attr-defined]

    def test_no_thinking_config_when_budget_is_unset(self):
        llm, models = gemini_with({"gemini-2.5-flash": text("ok")})

        llm.generate("p")

        assert models.calls[0]["config"].thinking_config is None  # type: ignore[attr-defined]

    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="at least one"):
            GeminiLLM(api_key="k", models=[])


class TestSettings:
    def test_model_list_comes_from_environment(self, monkeypatch):
        monkeypatch.setenv("CITEWISE_LLM_MODELS", " gemini-x , ,gemini-y ")

        assert load_settings().llm_models == ("gemini-x", "gemini-y")

    def test_default_models_when_environment_is_unset(self, monkeypatch):
        monkeypatch.delenv("CITEWISE_LLM_MODELS", raising=False)

        assert load_settings().llm_models == DEFAULT_LLM_MODELS


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
