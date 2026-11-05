from citewise.indexing import index_directory
from citewise.store import ChunkStore

from tests.pdf_factory import make_pdf_bytes

MAX_CHARS = 500


def stored_texts(store: ChunkStore) -> list[str]:
    return [chunk.text for chunk in store.all_chunks()]


def test_replacing_a_pdf_removes_the_old_versions_chunks(tmp_path, keyword_embedder):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store = ChunkStore(None, "replaced-pdf", keyword_embedder)
    (data_dir / "notes.pdf").write_bytes(make_pdf_bytes("Old sql notes."))
    index_directory(data_dir, store, MAX_CHARS)

    (data_dir / "notes.pdf").write_bytes(make_pdf_bytes("New chart notes."))
    index_directory(data_dir, store, MAX_CHARS)

    assert stored_texts(store) == ["New chart notes."]


def test_deleting_a_pdf_removes_its_chunks(tmp_path, keyword_embedder):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store = ChunkStore(None, "deleted-pdf", keyword_embedder)
    (data_dir / "keep.pdf").write_bytes(make_pdf_bytes("Keep these sql notes."))
    (data_dir / "gone.pdf").write_bytes(make_pdf_bytes("Remove these chart notes."))
    index_directory(data_dir, store, MAX_CHARS)

    (data_dir / "gone.pdf").unlink()
    index_directory(data_dir, store, MAX_CHARS)

    assert stored_texts(store) == ["Keep these sql notes."]


def test_reindexing_unchanged_pdfs_keeps_their_chunks(tmp_path, keyword_embedder):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store = ChunkStore(None, "unchanged-pdf", keyword_embedder)
    (data_dir / "notes.pdf").write_bytes(make_pdf_bytes("Sql notes."))
    index_directory(data_dir, store, MAX_CHARS)

    added = index_directory(data_dir, store, MAX_CHARS)

    assert added == {"notes.pdf": 0}
    assert stored_texts(store) == ["Sql notes."]
