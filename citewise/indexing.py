import logging
from pathlib import Path

from citewise.chunking import chunk_document
from citewise.ingest import Document, load_pdf
from citewise.store import ChunkStore

logger = logging.getLogger(__name__)


def index_pdf(doc: Document, store: ChunkStore, max_chars: int) -> int:
    """Index one document and return how many chunks were added (0 if it was already indexed)."""
    if store.has_document(doc.doc_id):
        return 0
    empty_pages = [p.number for p in doc.pages if not p.text]
    if empty_pages:
        logger.warning("%s: no extractable text on pages %s (scanned?)", doc.name, empty_pages)
    chunks = chunk_document(doc, max_chars)
    store.add(chunks)
    return len(chunks)


def index_directory(data_dir: Path, store: ChunkStore, max_chars: int) -> dict[str, int]:
    added = {pdf.name: index_pdf(load_pdf(pdf), store, max_chars) for pdf in sorted(data_dir.glob("*.pdf"))}
    logger.info("indexed %s", added)
    return added
