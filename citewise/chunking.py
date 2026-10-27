import re
from dataclasses import dataclass

from citewise.ingest import Document

# Also splits "end.Next" because pypdf glues lines and repeated paragraphs together without a space.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s*(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    text: str


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_BOUNDARY.split(text) if s.strip()]


def _normalise(sentence: str) -> str:
    return " ".join(sentence.lower().split())


def _unique_sentences(sentences: list[str], seen: set[str]) -> list[str]:
    kept: list[str] = []
    kept_keys: list[str] = []
    for sentence in sentences:
        key = _normalise(sentence)
        # A repeat can lose a leading heading ("Page 1: Intro ..."), so also drop sentences contained in a kept one.
        if key in seen or any(key in earlier for earlier in kept_keys):
            continue
        seen.add(key)
        kept.append(sentence)
        kept_keys.append(key)
    return kept


def _pack(sentences: list[str], max_chars: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join([*current, sentence])) > max_chars:
            groups.append(" ".join(current))
            # Carry the last sentence forward so a fact split across chunks keeps its context.
            current = [current[-1]]
            if len(" ".join([*current, sentence])) > max_chars:
                current = []
        current.append(sentence)
    if current:
        groups.append(" ".join(current))
    return groups


def chunk_document(doc: Document, max_chars: int) -> list[Chunk]:
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for page in doc.pages:
        sentences = _unique_sentences(split_sentences(page.text), seen)
        for index, text in enumerate(_pack(sentences, max_chars)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:p{page.number}:c{index}",
                    doc_id=doc.doc_id,
                    doc_name=doc.name,
                    page=page.number,
                    text=text,
                )
            )
    return chunks
