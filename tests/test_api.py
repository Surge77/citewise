import pytest
from fastapi.testclient import TestClient

from citewise.answer import Answer, Citation
from citewise.api import create_app
from citewise.library import DocumentInfo, InvalidPDFError, UploadResult
from citewise.llm import LLMError

UPLOAD = UploadResult(
    doc_id="abc123", name="notes.pdf", pages=3, chunks_added=5, already_indexed=False, empty_pages=[]
)


class StubService:
    def __init__(self, result: Answer | Exception, upload: UploadResult | Exception = UPLOAD):
        self.result = result
        self.upload = upload
        self.uploads: list[tuple[str, bytes]] = []

    def ask(self, question: str) -> Answer:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def add_pdf(self, name: str, data: bytes) -> UploadResult:
        self.uploads.append((name, data))
        if isinstance(self.upload, Exception):
            raise self.upload
        return self.upload

    def documents(self) -> list[DocumentInfo]:
        return [DocumentInfo(doc_id="abc123", name="notes.pdf", chunks=5)]


def client_for(result: Answer | Exception, **kwargs) -> TestClient:
    return TestClient(create_app(lambda: StubService(result, **kwargs), max_upload_bytes=1_000))


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


class TestUpload:
    def test_new_document_returns_201_with_index_result(self):
        with client_for(ANSWER) as client:
            response = client.post("/api/documents?name=notes.pdf", content=b"%PDF-1.4 tiny")

        assert response.status_code == 201
        assert response.json()["data"]["chunks_added"] == 5

    def test_passes_name_and_raw_bytes_to_the_service(self):
        service = StubService(ANSWER)
        with TestClient(create_app(lambda: service, max_upload_bytes=1_000)) as client:
            client.post("/api/documents?name=my notes.pdf", content=b"%PDF-1.4 bytes")

        assert service.uploads == [("my notes.pdf", b"%PDF-1.4 bytes")]

    def test_already_indexed_document_returns_200(self):
        repeat = UploadResult(doc_id="abc", name="n.pdf", pages=1, chunks_added=0, already_indexed=True, empty_pages=[])
        with client_for(ANSWER, upload=repeat) as client:
            response = client.post("/api/documents?name=n.pdf", content=b"%PDF-1.4")

        assert response.status_code == 200
        assert response.json()["data"]["already_indexed"] is True

    def test_rejects_uploads_over_the_size_limit(self):
        with client_for(ANSWER) as client:
            response = client.post("/api/documents?name=big.pdf", content=b"%PDF-" + b"x" * 2_000)

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "FILE_TOO_LARGE"

    def test_invalid_pdf_returns_400_with_its_message(self):
        with client_for(ANSWER, upload=InvalidPDFError("This file is not a PDF.")) as client:
            response = client.post("/api/documents?name=x.pdf", content=b"MZ")

        assert response.status_code == 400
        assert response.json()["error"] == {"code": "INVALID_PDF", "message": "This file is not a PDF."}

    def test_empty_body_is_rejected(self):
        with client_for(ANSWER) as client:
            response = client.post("/api/documents?name=x.pdf", content=b"")

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_PDF"

    def test_missing_name_is_a_validation_error(self):
        with client_for(ANSWER) as client:
            response = client.post("/api/documents", content=b"%PDF-1.4")

        assert response.status_code == 422
        assert response.json()["error"]["fields"][0]["field"] == "name"

    def test_unexpected_upload_failure_returns_generic_500(self):
        with client_for(ANSWER, upload=OSError("disk full at C:/secret")) as client:
            response = client.post("/api/documents?name=x.pdf", content=b"%PDF-1.4")

        assert response.status_code == 500
        assert "secret" not in response.text


def test_lists_indexed_documents():
    with client_for(ANSWER) as client:
        response = client.get("/api/documents")

    assert response.json()["data"] == [{"doc_id": "abc123", "name": "notes.pdf", "chunks": 5}]
