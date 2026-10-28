import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from citewise.llm import LLM
from citewise.store import Hit

NOT_FOUND = "I couldn't find this in your documents."
SNIPPET_CHARS = 240
CITATION_MARK = re.compile(r"\[(\d+)\]")

PROMPT_TEMPLATE = """You answer questions using only the numbered sources below.

Rules:
- Support every claim with the number of its source in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, reply with exactly this sentence and nothing else: {not_found}
- Do not use outside knowledge. Keep the answer under 120 words.

Sources:
{sources}

Question: {question}
Answer:"""


class Searcher(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


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


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(
        f"[{n}] ({hit.chunk.doc_name}, p. {hit.chunk.page}) {hit.chunk.text}" for n, hit in enumerate(hits, start=1)
    )
    return PROMPT_TEMPLATE.format(not_found=NOT_FOUND, sources=sources, question=question)


def cited_numbers(text: str, source_count: int) -> list[int]:
    numbers = {int(match) for match in CITATION_MARK.findall(text)}
    return sorted(n for n in numbers if 1 <= n <= source_count)


def _snippet(text: str) -> str:
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + "…"


class Answerer:
    def __init__(self, searcher: Searcher, llm: LLM, top_k: int, min_similarity: float) -> None:
        self._searcher = searcher
        self._llm = llm
        self._top_k = top_k
        self._min_similarity = min_similarity

    def ask(self, question: str) -> Answer:
        question = question.strip()
        hits = self._relevant_hits(question)
        if not hits:
            return REFUSAL
        return self._finish(self._llm.generate(build_prompt(question, hits)), hits)

    def ask_stream(self, question: str) -> Iterator[Delta | Done]:
        question = question.strip()
        hits = self._relevant_hits(question)
        if not hits:
            yield Done(REFUSAL)
            return
        parts: list[str] = []
        for text in self._llm.stream(build_prompt(question, hits)):
            parts.append(text)
            yield Delta(text)
        yield Done(self._finish("".join(parts), hits))

    def _relevant_hits(self, question: str) -> list[Hit]:
        hits = self._searcher.search(question, self._top_k)
        # A question-level gate: keyword-only hits have similarity 0.0 but are kept when meaning search agrees.
        if not hits or max(h.similarity for h in hits) < self._min_similarity:
            return []
        return hits

    def _finish(self, raw_text: str, hits: list[Hit]) -> Answer:
        text = raw_text.strip()
        if NOT_FOUND in text:
            return REFUSAL
        citations = [
            Citation(
                number=n,
                doc_name=hits[n - 1].chunk.doc_name,
                page=hits[n - 1].chunk.page,
                snippet=_snippet(hits[n - 1].chunk.text),
            )
            for n in cited_numbers(text, len(hits))
        ]
        return Answer(text=text, citations=citations, found=True)
