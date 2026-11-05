import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from citewise.answer import Answer
from citewise.config import PROJECT_ROOT
from citewise.llm import LLMError
from citewise.service import build_answerer

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500


class AnswerService(Protocol):
    def ask(self, question: str) -> Answer: ...


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ok(data: Any) -> JSONResponse:
    return JSONResponse({"success": True, "data": data, "timestamp": _now()})


def _error(status: int, code: str, message: str, fields: list[dict[str, str]] | None = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if fields is not None:
        error["fields"] = fields
    return JSONResponse({"success": False, "error": error, "timestamp": _now()}, status_code=status)


def create_app(answerer_factory: Callable[[], AnswerService] = build_answerer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Loading the model and indexing PDFs is CPU-bound; keep it off the event loop.
        app.state.answerer = await asyncio.get_running_loop().run_in_executor(None, answerer_factory)
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
        answerer: AnswerService = request.app.state.answerer
        try:
            answer = await asyncio.get_running_loop().run_in_executor(None, answerer.ask, body.question)
        except LLMError as exc:
            logger.warning("llm unavailable: %s", exc)
            return _error(502, "LLM_UNAVAILABLE", "The answer service is unavailable. Try again shortly.")
        except Exception:
            logger.exception("unexpected error while answering")
            return _error(500, "INTERNAL_ERROR", "Something went wrong.")
        return _ok({"answer": answer.text, "found": answer.found, "citations": [asdict(c) for c in answer.citations]})

    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "static", html=True), name="static")
    return app


app = create_app()
