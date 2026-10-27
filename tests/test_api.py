import pytest
from fastapi.testclient import TestClient

from citewise.answer import Answer, Citation
from citewise.api import create_app
from citewise.llm import LLMError


class StubAnswerer:
    def __init__(self, result: Answer | Exception):
        self.result = result

    def ask(self, question: str) -> Answer:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def client_for(result: Answer | Exception) -> TestClient:
    return TestClient(create_app(lambda: StubAnswerer(result)))


ANSWER = Answer(
    text="SQL joins tables [1].",
    citations=[Citation(number=1, doc_name="notes.pdf", page=7, snippet="SQL is used to join.")],
    found=True,
)


def test_ask_returns_answer_and_citations_in_success_envelope():
    with client_for(ANSWER) as client:
        response = client.post("/api/ask", json={"question": "What is SQL?"})

    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True
    assert body["data"]["answer"] == "SQL joins tables [1]."
    assert body["data"]["citations"][0] == {
        "number": 1,
        "doc_name": "notes.pdf",
        "page": 7,
        "snippet": "SQL is used to join.",
    }
    assert "timestamp" in body


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 501}])
def test_rejects_invalid_questions_with_field_errors(payload):
    with client_for(ANSWER) as client:
        response = client.post("/api/ask", json=payload)

    body = response.json()
    assert response.status_code == 422
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["fields"][0]["field"] == "question"


def test_llm_failure_returns_502_without_leaking_details():
    with client_for(LLMError("quota exceeded for key AIza-secret")) as client:
        response = client.post("/api/ask", json={"question": "q"})

    body = response.json()
    assert response.status_code == 502
    assert body["error"]["code"] == "LLM_UNAVAILABLE"
    assert "AIza" not in response.text


def test_unexpected_failure_returns_generic_500():
    with client_for(RuntimeError("C:/Users/someone/private/path")) as client:
        response = client.post("/api/ask", json={"question": "q"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "private" not in response.text


def test_health_reports_ok():
    with client_for(ANSWER) as client:
        response = client.get("/api/health")

    assert response.json()["data"] == {"status": "ok"}


def test_serves_the_chat_page():
    with client_for(ANSWER) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Citewise" in response.text
