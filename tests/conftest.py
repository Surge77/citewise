from dataclasses import dataclass, field

import pytest

from citewise.chunking import Chunk
from citewise.ingest import Document, Page
from citewise.store import Hit


def make_document(*page_texts: str, name: str = "notes.pdf", doc_id: str = "doc1") -> Document:
    pages = tuple(Page(number=i, text=text) for i, text in enumerate(page_texts, start=1))
    return Document(doc_id=doc_id, name=name, pages=pages)


def make_hit(text: str, page: int = 1, similarity: float = 0.8, doc_name: str = "notes.pdf") -> Hit:
    chunk = Chunk(chunk_id=f"doc1:p{page}:c0", doc_id="doc1", doc_name=doc_name, page=page, text=text)
    return Hit(chunk=chunk, similarity=similarity)


class KeywordEmbedder:
    """Deterministic stand-in for MiniLM: one dimension per vocabulary word, L2-normalised."""

    VOCAB = ("sql", "join", "cleaning", "missing", "chart", "visualization", "bread", "model")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = text.lower().split()
            raw = [float(sum(w.startswith(v) for w in words)) for v in self.VOCAB] + [0.01]
            norm = sum(x * x for x in raw) ** 0.5
            vectors.append([x / norm for x in raw])
        return vectors


@dataclass
class FakeSearcher:
    hits: list[Hit]
    queries: list[str] = field(default_factory=list)

    def search(self, query: str, k: int) -> list[Hit]:
        self.queries.append(query)
        return self.hits[:k]


@dataclass
class FakeLLM:
    reply: str
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture
def keyword_embedder() -> KeywordEmbedder:
    return KeywordEmbedder()
