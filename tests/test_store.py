from citewise.chunking import Chunk
from citewise.indexing import index_pdf
from citewise.store import ChunkStore

from tests.conftest import make_document


def make_chunk(text: str, page: int, idx: int = 0) -> Chunk:
    return Chunk(chunk_id=f"doc1:p{page}:c{idx}", doc_id="doc1", doc_name="notes.pdf", page=page, text=text)


def test_search_returns_most_similar_chunk_first_with_metadata(tmp_path, keyword_embedder):
    store = ChunkStore(tmp_path / "db", "test", keyword_embedder)
    store.add([make_chunk("sql join tables", page=7), make_chunk("chart visualization", page=5)])

    hits = store.search("sql join", k=2)

    assert hits[0].chunk.page == 7
    assert hits[0].chunk.doc_name == "notes.pdf"
    assert hits[0].similarity > hits[1].similarity


def test_search_on_empty_store_returns_nothing(tmp_path, keyword_embedder):
    store = ChunkStore(tmp_path / "db", "test", keyword_embedder)

    assert store.search("anything", k=3) == []


def test_adding_same_chunk_ids_twice_does_not_duplicate(tmp_path, keyword_embedder):
    store = ChunkStore(tmp_path / "db", "test", keyword_embedder)
    chunks = [make_chunk("sql join", page=1)]

    store.add(chunks)
    store.add(chunks)

    assert store.count() == 1


def test_all_chunks_returns_every_stored_chunk_with_metadata(tmp_path, keyword_embedder):
    store = ChunkStore(tmp_path / "db", "test", keyword_embedder)
    store.add([make_chunk("sql", page=1), make_chunk("chart", page=2)])

    assert sorted((c.page, c.text, c.doc_name) for c in store.all_chunks()) == [
        (1, "sql", "notes.pdf"),
        (2, "chart", "notes.pdf"),
    ]


def test_in_memory_store_works_without_a_directory(keyword_embedder):
    store = ChunkStore(None, "in-memory-test", keyword_embedder)
    store.add([make_chunk("sql join", page=3)])

    assert store.search("sql", k=1)[0].chunk.page == 3


def test_has_document_reports_indexed_documents(tmp_path, keyword_embedder):
    store = ChunkStore(tmp_path / "db", "test", keyword_embedder)
    store.add([make_chunk("sql", page=1)])

    assert store.has_document("doc1") is True
    assert store.has_document("other") is False


def test_index_pdf_skips_documents_already_indexed(tmp_path, keyword_embedder):
    store = ChunkStore(tmp_path / "db", "test", keyword_embedder)
    doc = make_document("Sql joins tables. Charts help.")

    first = index_pdf(doc, store, max_chars=500)
    second = index_pdf(doc, store, max_chars=500)

    assert first == 1
    assert second == 0
