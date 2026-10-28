import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Protocol

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from citewise.answer import Answer
from citewise.config import PROJECT_ROOT
from citewise.library import DocumentInfo, InvalidPDFError, UploadResult
from citewise.llm import LLMError
from citewise.service import build_service

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_FILENAME_CHARS = 255


class Service(Protocol):
    def ask(self, question: str) -> Answer: ...
    def add_pdf(self, name: str, data: bytes) -> UploadResult: ...
    def documents(self) -> list[DocumentInfo]: ...


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class UploadTooLargeError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ok(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse({"success": True, "data": data, "timestamp": _now()}, status_code=status)


def _error(status: int, code: str, message: str, fields: list[dict[str, str]] | None = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if fields is not None:
        error["fields"] = fields
    return JSONResponse({"success": False, "error": error, "timestamp": _now()}, status_code=status)


async def _read_body(request: Request, limit: int) -> bytes:
    """Read the request body, stopping as soon as it exceeds the limit instead of buffering all of it."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise UploadTooLargeError
    body = bytearray()
    async for part in request.stream():
        body += part
        if len(body) > limit:
            raise UploadTooLargeError
    return bytes(body)


def create_app(
    service_factory: Callable[[], Service] = build_service, max_upload_bytes: int = MAX_UPLOAD_BYTES
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Loading models and indexing PDFs is CPU-bound; keep it off the event loop.
        app.state.service = await asyncio.get_running_loop().run_in_executor(None, service_factory)
        yield

    app = FastAPI(title="Citewise", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [{"field": str(err["loc"][-1]), "message": err["msg"]} for err in exc.errors()]
        return _error(422, "VALIDATION_ERROR", "Invalid request.", fields)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return _ok({"status": "ok"})

    @app.post("/api/ask")
    async def ask(body: AskRequest, request: Request) -> JSONResponse:
        service: Service = request.app.state.service
        try:
            answer = await asyncio.get_running_loop().run_in_executor(None, service.ask, body.question)
        except LLMError as exc:
            logger.warning("llm unavailable: %s", exc)
            return _error(502, "LLM_UNAVAILABLE", "The answer service is unavailable. Try again shortly.")
        except Exception:
            logger.exception("unexpected error while answering")
            return _error(500, "INTERNAL_ERROR", "Something went wrong.")
        return _ok({"answer": answer.text, "found": answer.found, "citations": [asdict(c) for c in answer.citations]})

    @app.post("/api/documents")
    async def upload(
        request: Request, name: str = Query(min_length=1, max_length=MAX_FILENAME_CHARS)
    ) -> JSONResponse:
        """Upload one PDF as the raw request body (Content-Type: application/pdf), with its file name as ?name=."""
        service: Service = request.app.state.service
        try:
            data = await _read_body(request, max_upload_bytes)
        except UploadTooLargeError:
            limit_mb = max_upload_bytes / (1024 * 1024)
            return _error(413, "FILE_TOO_LARGE", f"PDFs can be at most {limit_mb:g} MB.")
        if not data:
            return _error(400, "INVALID_PDF", "The uploaded file is empty.")
        try:
            result = await asyncio.get_running_loop().run_in_executor(None, service.add_pdf, name, data)
        except InvalidPDFError as exc:
            return _error(400, "INVALID_PDF", str(exc))
        except Exception:
            logger.exception("unexpected error while indexing an upload")
            return _error(500, "INTERNAL_ERROR", "Something went wrong.")
        return _ok(asdict(result), status=200 if result.already_indexed else 201)

    @app.get("/api/documents")
    async def documents(request: Request) -> JSONResponse:
        service: Service = request.app.state.service
        docs = await asyncio.get_running_loop().run_in_executor(None, service.documents)
        return _ok([asdict(d) for d in docs])

    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "static", html=True), name="static")
    return app


app = create_app()
