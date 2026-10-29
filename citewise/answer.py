import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from citewise.chunking import Chunk
from citewise.llm import LLM
from citewise.pages import describe_library, page_sources
from citewise.store import Hit

NOT_FOUND = "I couldn't find this in your documents."
SNIPPET_CHARS = 240
CITATION_MARK = re.compile(r"\[(\d+)\]")

WHOLE_LIBRARY_SCOPE = (
    "The sources are the complete text of every document in the user's library, one source per page. "
    "The library contains: {library}."
)
PASSAGES_SCOPE = "The sources are the passages from the user's documents that best match the question."

PROMPT_TEMPLATE = """You are Citewise. You answer questions about the user's documents using only the numbered sources below.
{scope}

Rules:
- Support every claim with the number of its source in square brackets, like [1] or [2][3].
- You may summarise, compare, list, count or explain across the sources whenever the question asks for it.
- If the sources do not contain what the question asks about, reply with exactly this sentence and nothing else: {not_found}
- Do not use outside knowledge. Be concise: about 200 words at most, unless the question asks for a summary or a list.
{conversation}
Sources:
{sources}

Question: {question}
Answer:"""


class Searcher(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


class ChunkSource(Protocol):
    def all_chunks(self) -> list[Chunk]: ...


@dataclass(frozen=True)
class Turn:
    """One earlier message, so follow-ups like "explain that more simply" have something to refer to."""

    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class Citation:
    number: int
    doc_name: str
    page: int
    snippet: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    found: bool


@dataclass(frozen=True)
class Delta:
    """A piece of answer text as the model produces it."""

    text: str


@dataclass(frozen=True)
class Done:
    """The final answer; citations are parsed here because a [n] marker can be split across deltas."""

    answer: Answer


REFUSAL = Answer(text=NOT_FOUND, citations=[], found=False)


def _conversation_block(history: Sequence[Turn]) -> str:
    if not history:
        return ""
    lines = "\n".join(f"{'User' if t.role == 'user' else 'Assistant'}: {t.text}" for t in history)
    return f"\nConversation so far (context for follow-up questions; cite only the sources):\n{lines}\n"


def build_prompt(
    question: str, hits: list[Hit], scope: str = PASSAGES_SCOPE, history: Sequence[Turn] = ()
) -> str:
    sources = "\n\n".join(
        f"[{n}] ({hit.chunk.doc_name}, p. {hit.chunk.page}) {hit.chunk.text}" for n, hit in enumerate(hits, start=1)
    )
    return PROMPT_TEMPLATE.format(
        scope=scope,
        not_found=NOT_FOUND,
        conversation=_conversation_block(history),
        sources=sources,
        question=question,
    )


def cited_numbers(text: str, source_count: int) -> list[int]:
    numbers = {int(match) for match in CITATION_MARK.findall(text)}
    return sorted(n for n in numbers if 1 <= n <= source_count)


def _snippet(text: str) -> str:
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + "…"


class Answerer:
    """Answers from the whole library when it fits in one prompt, otherwise from the best-matching passages.

    Whole-library mode handles questions that no single passage matches: summaries, overviews, lists, counts and
    comparisons across documents. Search mode keeps large libraries within the model's limits.
    """

    def __init__(
        self,
        searcher: Searcher,
        llm: LLM,
        top_k: int,
        min_similarity: float,
        library: ChunkSource | None = None,
        full_context_chars: int = 0,
    ) -> None:
        self._searcher = searcher
        self._llm = llm
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._library = library
        self._full_context_chars = full_context_chars

    def ask(self, question: str, history: Sequence[Turn] = ()) -> Answer:
        prepared = self._prepare(question.strip(), history)
        if prepared is None:
            return REFUSAL
        prompt, sources = prepared
        return self._finish(self._llm.generate(prompt), sources)

    def ask_stream(self, question: str, history: Sequence[Turn] = ()) -> Iterator[Delta | Done]:
        prepared = self._prepare(question.strip(), history)
        if prepared is None:
            yield Done(REFUSAL)
            return
        prompt, sources = prepared
        parts: list[str] = []
        for text in self._llm.stream(prompt):
            parts.append(text)
            yield Delta(text)
        yield Done(self._finish("".join(parts), sources))

    def _prepare(self, question: str, history: Sequence[Turn]) -> tuple[str, list[Hit]] | None:
        pages = self._whole_library()
        if pages is not None:
            if not pages:
                return None
            scope = WHOLE_LIBRARY_SCOPE.format(library=describe_library(pages))
            return build_prompt(question, pages, scope, history), pages
        hits = self._relevant_hits(self._search_query(question, history))
        if not hits:
            return None
        return build_prompt(question, hits, PASSAGES_SCOPE, history), hits

    def _whole_library(self) -> list[Hit] | None:
        """Every page, or None when there is no library or it is too large to send in one prompt."""
        if self._library is None or self._full_context_chars <= 0:
            return None
        pages = page_sources(self._library.all_chunks())
        if sum(len(p.chunk.text) for p in pages) > self._full_context_chars:
            return None
        return pages

    @staticmethod
    def _search_query(question: str, history: Sequence[Turn]) -> str:
        # A follow-up like "give an example" only finds the right passage alongside the question it follows.
        previous = [t.text for t in history if t.role == "user"][-1:]
        return " ".join([*previous, question])

    def _relevant_hits(self, query: str) -> list[Hit]:
        hits = self._searcher.search(query, self._top_k)
        # A question-level gate: keyword-only hits have similarity 0.0 but are kept when meaning search agrees.
        if not hits or max(h.similarity for h in hits) < self._min_similarity:
            return []
        return hits

    def _finish(self, raw_text: str, sources: list[Hit]) -> Answer:
        text = raw_text.strip()
        if NOT_FOUND in text:
            return REFUSAL
        citations = [
            Citation(
                number=n,
                doc_name=sources[n - 1].chunk.doc_name,
                page=sources[n - 1].chunk.page,
                snippet=_snippet(sources[n - 1].chunk.text),
            )
            for n in cited_numbers(text, len(sources))
        ]
        return Answer(text=text, citations=citations, found=True)
