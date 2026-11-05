import json
import pytest
from fastapi.testclient import TestClient

from citewise.answer import NOT_FOUND, Answer, Answerer, Citation, Delta, Done
from citewise.api import RATE_LIMITED_MESSAGE, create_app
from citewise.llm import LLMError, LLMRateLimitedError

from tests.conftest import FakeLLM, FakeSearcher, make_hit
from tests.test_api import StubService


def stream_answerer(hits, reply="SQL joins tables [1]. Charts help [2].", min_similarity=0.3):
    llm = FakeLLM(reply=reply)
    return Answerer(searcher=FakeSearcher(hits=hits), llm=llm, top_k=4, min_similarity=min_similarity), llm


class TestAnswererStream:
    def test_yields_text_deltas_then_done_with_citations(self):
        hits = [make_hit("SQL joins tables.", page=7), make_hit("Charts help.", page=5)]
        answerer, _ = stream_answerer(hits)

        events = list(answerer.ask_stream("How do I join?"))

        deltas = [e.text for e in events if isinstance(e, Delta)]
        done = events[-1]
        assert len(deltas) > 1
        assert "".join(deltas) == "SQL joins tables [1]. Charts help [2]."
        assert isinstance(done, Done)
        assert done.answer.found is True
        assert [(c.number, c.page) for c in done.answer.citations] == [(1, 7), (2, 5)]

    def test_refusal_is_a_single_done_event_without_calling_the_llm(self):
        answerer, llm = stream_answerer([make_hit("Unrelated.", similarity=0.05)])

        events = list(answerer.ask_stream("How do I bake bread?"))

        assert len(events) == 1
        assert isinstance(events[0], Done)
        assert events[0].answer == Answer(text=NOT_FOUND, citations=[], found=False)
        assert llm.prompts == []

    def test_model_refusal_ends_with_not_found(self):
        answerer, _ = stream_answerer([make_hit("Some text.")], reply=NOT_FOUND)

        done = list(answerer.ask_stream("q"))[-1]

        assert isinstance(done, Done)
        assert done.answer.found is False


class StreamingStub(StubService):
    def __init__(self, events: list[Delta | Done] | Exception, fail_after: int | None = None):
        super().__init__(result=Answer(text="", citations=[], found=False))
        self.events = events
        self.fail_after = fail_after

    def ask_stream(self, question: str):
        if isinstance(self.events, Exception):
            raise self.events
        for index, event in enumerate(self.events):
            if self.fail_after is not None and index == self.fail_after:
                raise LLMError("connection dropped, key AIza-secret")
            yield event


def stream_lines(stub: StreamingStub, question: str = "What is SQL?") -> list[dict]:
    with TestClient(create_app(lambda: stub)) as client:
        response = client.post("/api/ask/stream", json={"question": question})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines() if line]


DONE = Done(
    Answer(
        text="SQL joins [1].",
        citations=[Citation(number=1, doc_name="notes.pdf", page=7, snippet="SQL joins.")],
        found=True,
    )
)


class TestStreamEndpoint:
    def test_streams_delta_lines_then_a_done_line_with_citations(self):
        lines = stream_lines(StreamingStub([Delta("SQL "), Delta("joins [1]."), DONE]))

        assert lines[:2] == [{"type": "delta", "text": "SQL "}, {"type": "delta", "text": "joins [1]."}]
        assert lines[2]["type"] == "done"
        assert lines[2]["answer"] == "SQL joins [1]."
        assert lines[2]["citations"][0]["page"] == 7

    def test_llm_failure_mid_stream_ends_with_error_line_without_details(self):
        lines = stream_lines(StreamingStub([Delta("SQL "), Delta("joins"), DONE], fail_after=1))

        assert lines[0] == {"type": "delta", "text": "SQL "}
        assert lines[-1]["type"] == "error"
        assert lines[-1]["code"] == "LLM_UNAVAILABLE"
        assert "AIza" not in json.dumps(lines)

    def test_rate_limit_before_any_text_returns_rate_limited_line(self):
        lines = stream_lines(StreamingStub(LLMRateLimitedError("all models out of quota")))

        assert lines == [{"type": "error", "code": "RATE_LIMITED", "message": RATE_LIMITED_MESSAGE}]

    def test_unexpected_failure_before_any_text_returns_generic_error_line(self):
        lines = stream_lines(StreamingStub(RuntimeError("C:/private/path")))

        assert lines == [{"type": "error", "code": "INTERNAL_ERROR", "message": "Something went wrong."}]

    def test_invalid_question_is_rejected_before_streaming(self):
        with TestClient(create_app(lambda: StreamingStub([DONE]))) as client:
            response = client.post("/api/ask/stream", json={"question": "  "})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"
