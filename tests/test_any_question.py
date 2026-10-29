from dataclasses import dataclass

from citewise.answer import NOT_FOUND, Answerer, Done, Turn
from citewise.chunking import Chunk

from tests.conftest import FakeLLM, FakeSearcher, make_hit


@dataclass
class FakeLibrary:
    chunks: list[Chunk]

    def all_chunks(self) -> list[Chunk]:
        return self.chunks


def page(doc: str, number: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"{doc}:p{number}:c0", doc_id=doc, doc_name=doc, page=number, text=text)


LIBRARY = FakeLibrary(
    [
        page("notes.pdf", 1, "Analytics has four types."),
        page("notes.pdf", 2, "SQL uses HAVING to filter groups."),
        page("resume.pdf", 1, "Built a FastAPI service."),
    ]
)


def answerer_for(reply: str, library: FakeLibrary | None = LIBRARY, full_context_chars: int = 10_000, hits=None):
    llm = FakeLLM(reply=reply)
    searcher = FakeSearcher(hits=hits if hits is not None else [make_hit("Searched passage.", similarity=0.9)])
    answerer = Answerer(
        searcher=searcher,
        llm=llm,
        top_k=4,
        min_similarity=0.3,
        library=library,
        full_context_chars=full_context_chars,
    )
    return answerer, llm, searcher


class TestWholeLibraryMode:
    def test_sends_every_page_of_every_document_as_numbered_sources(self):
        answerer, llm, searcher = answerer_for("Summary [1][2][3].")

        answerer.ask("Summarize my documents")

        prompt = llm.prompts[0]
        assert "[1] (notes.pdf, p. 1) Analytics has four types." in prompt
        assert "[2] (notes.pdf, p. 2) SQL uses HAVING to filter groups." in prompt
        assert "[3] (resume.pdf, p. 1) Built a FastAPI service." in prompt
        assert "notes.pdf (2 pages), resume.pdf (1 page)" in prompt
        assert searcher.queries == []

    def test_answers_questions_that_match_no_passage_like_summaries(self):
        # No similarity gate in this mode: "summarize" shares no words with any page.
        answerer, _, _ = answerer_for("The notes cover analytics [1] and SQL [2].", hits=[])

        answer = answerer.ask("summarize the pdf")

        assert answer.found is True
        assert [(c.doc_name, c.page) for c in answer.citations] == [("notes.pdf", 1), ("notes.pdf", 2)]

    def test_off_topic_questions_are_still_refused_by_the_model(self):
        answerer, _, _ = answerer_for(NOT_FOUND)

        assert answerer.ask("Who won the World Cup?").found is False

    def test_empty_library_refuses_without_calling_the_model(self):
        answerer, llm, _ = answerer_for("unused", library=FakeLibrary([]))

        assert answerer.ask("summarize").found is False
        assert llm.prompts == []

    def test_falls_back_to_search_when_the_library_is_too_large(self):
        answerer, llm, searcher = answerer_for("Found it [1].", full_context_chars=10)

        answer = answerer.ask("What is HAVING?")

        assert searcher.queries == ["What is HAVING?"]
        assert "Searched passage." in llm.prompts[0]
        assert answer.found is True

    def test_streaming_uses_the_same_whole_library_sources(self):
        answerer, llm, _ = answerer_for("Overview [1].")

        events = list(answerer.ask_stream("What is in my library?"))

        assert isinstance(events[-1], Done)
        assert "[3] (resume.pdf, p. 1)" in llm.prompts[0]


class TestFollowUps:
    def test_recent_conversation_is_included_in_the_prompt(self):
        answerer, llm, _ = answerer_for("Simpler [2].")
        history = [Turn(role="user", text="What is HAVING?"), Turn(role="assistant", text="It filters groups [2].")]

        answerer.ask("Explain that more simply", history)

        prompt = llm.prompts[0]
        assert "User: What is HAVING?" in prompt
        assert "Assistant: It filters groups [2]." in prompt
        assert prompt.index("Assistant: It filters groups") < prompt.index("Question: Explain that more simply")

    def test_no_conversation_section_without_history(self):
        answerer, llm, _ = answerer_for("Answer [1].")

        answerer.ask("What is HAVING?")

        assert "Conversation so far" not in llm.prompts[0]

    def test_search_mode_adds_the_previous_question_to_the_search_query(self):
        answerer, _, searcher = answerer_for("Answer [1].", full_context_chars=10)
        history = [Turn(role="user", text="What is HAVING?"), Turn(role="assistant", text="It filters groups.")]

        answerer.ask("Give an example", history)

        assert searcher.queries == ["What is HAVING? Give an example"]
