from collections import defaultdict
from dataclasses import replace

from citewise.chunking import Chunk, split_sentences
from citewise.store import Hit

# Whole pages are complete context, not search matches; this keeps them past the similarity gate.
FULL_SIMILARITY = 1.0


def _chunk_index(chunk: Chunk) -> int:
    return int(chunk.chunk_id.rsplit(":c", 1)[-1])


def _join_page(chunks: list[Chunk]) -> str:
    """Join a page's chunks in order, dropping the sentence each chunk repeats from the one before it."""
    text = ""
    for chunk in sorted(chunks, key=_chunk_index):
        part = chunk.text
        previous = split_sentences(text)
        if previous and part.startswith(previous[-1]):
            part = part[len(previous[-1]) :].lstrip()
        text = f"{text} {part}".strip()
    return text


def page_sources(chunks: list[Chunk]) -> list[Hit]:
    """Rebuild every page of every document from stored chunks, one source per page."""
    by_page: defaultdict[tuple[str, str, int], list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_page[(chunk.doc_name, chunk.doc_id, chunk.page)].append(chunk)
    sources = []
    for (doc_name, doc_id, page), page_chunks in sorted(by_page.items()):
        first = page_chunks[0]
        page_chunk = replace(first, chunk_id=f"{doc_id}:p{page}", text=_join_page(page_chunks))
        sources.append(Hit(chunk=page_chunk, similarity=FULL_SIMILARITY))
    return sources


def describe_library(pages: list[Hit]) -> str:
    counts: dict[str, int] = {}
    for hit in pages:
        counts[hit.chunk.doc_name] = counts.get(hit.chunk.doc_name, 0) + 1
    return ", ".join(f"{name} ({n} page{'s' if n != 1 else ''})" for name, n in counts.items())
