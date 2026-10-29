from citewise.chunking import Chunk
from citewise.pages import describe_library, page_sources


def chunk(doc: str, page: int, index: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"{doc}-id:p{page}:c{index}", doc_id=f"{doc}-id", doc_name=doc, page=page, text=text)


class TestPageSources:
    def test_one_source_per_page_ordered_by_document_then_page(self):
        chunks = [chunk("b.pdf", 1, 0, "B one."), chunk("a.pdf", 2, 0, "A two."), chunk("a.pdf", 1, 0, "A one.")]

        pages = page_sources(chunks)

        assert [(h.chunk.doc_name, h.chunk.page, h.chunk.text) for h in pages] == [
            ("a.pdf", 1, "A one."),
            ("a.pdf", 2, "A two."),
            ("b.pdf", 1, "B one."),
        ]

    def test_joins_a_pages_chunks_in_order_without_the_overlapping_sentence(self):
        # The chunker repeats the last sentence of a chunk at the start of the next one.
        chunks = [chunk("a.pdf", 1, 1, "Second fact. Third fact."), chunk("a.pdf", 1, 0, "First fact. Second fact.")]

        assert page_sources(chunks)[0].chunk.text == "First fact. Second fact. Third fact."

    def test_sorts_chunk_indexes_numerically(self):
        chunks = [chunk("a.pdf", 1, i, f"Fact {i}.") for i in (10, 2, 1)]

        assert page_sources(chunks)[0].chunk.text == "Fact 1. Fact 2. Fact 10."

    def test_page_sources_carry_full_similarity_and_a_page_level_id(self):
        source = page_sources([chunk("a.pdf", 3, 0, "Text.")])[0]

        assert source.similarity == 1.0
        assert source.chunk.chunk_id == "a.pdf-id:p3"

    def test_empty_library_has_no_sources(self):
        assert page_sources([]) == []


def test_describe_library_lists_documents_with_page_counts():
    pages = page_sources([chunk("a.pdf", 1, 0, "x."), chunk("a.pdf", 2, 0, "y."), chunk("b.pdf", 1, 0, "z.")])

    assert describe_library(pages) == "a.pdf (2 pages), b.pdf (1 page)"
