import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from pypdf.errors import PdfReadError

from citewise.indexing import index_pdf
from citewise.ingest import parse_pdf
from citewise.store import ChunkStore

PDF_SIGNATURE = b"%PDF-"
UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._ -]")
MAX_STEM_CHARS = 100
DISAMBIGUATING_ID_CHARS = 8


class InvalidPDFError(ValueError):
    """The upload is not a readable PDF. The message is safe to show to users."""


@dataclass(frozen=True)
class UploadResult:
    doc_id: str
    name: str
    pages: int
    chunks_added: int
    already_indexed: bool
    empty_pages: list[int]


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    name: str
    chunks: int


def safe_filename(raw: str) -> str:
    """Reduce an uploaded name to a plain file name: no directories, no characters that mean anything to a shell."""
    base = raw.replace("\\", "/").rsplit("/", 1)[-1]
    stem = base[:-4] if base.lower().endswith(".pdf") else base
    stem = UNSAFE_FILENAME_CHARS.sub("_", stem).strip(" .")[:MAX_STEM_CHARS]
    return f"{stem or 'document'}.pdf"


class Library:
    def __init__(self, store: ChunkStore, data_dir: Path, max_chunk_chars: int) -> None:
        self._store = store
        self._data_dir = data_dir
        self._max_chunk_chars = max_chunk_chars
        # Serialises uploads so two requests can't claim the same file name or index the same document twice.
        self._lock = threading.Lock()

    def add_pdf(self, raw_name: str, data: bytes) -> UploadResult:
        if not data.startswith(PDF_SIGNATURE):
            raise InvalidPDFError("This file is not a PDF.")
        try:
            doc = parse_pdf(data, safe_filename(raw_name))
        except (PdfReadError, ValueError, KeyError, TypeError) as exc:
            raise InvalidPDFError("This PDF could not be read. It may be damaged or encrypted.") from exc

        empty_pages = [p.number for p in doc.pages if not p.text]
        with self._lock:
            if self._store.has_document(doc.doc_id):
                return UploadResult(doc.doc_id, doc.name, len(doc.pages), 0, True, empty_pages)
            target = self._unused_path(doc.name, doc.doc_id)
            doc = replace(doc, name=target.name)
            added = index_pdf(doc, self._store, self._max_chunk_chars)
            # Saved after indexing succeeds, so a failed upload leaves nothing behind in data/.
            target.write_bytes(data)
        return UploadResult(doc.doc_id, doc.name, len(doc.pages), added, False, empty_pages)

    def documents(self) -> list[DocumentInfo]:
        return [
            DocumentInfo(doc_id=doc_id, name=name, chunks=count)
            for (doc_id, name), count in sorted(self._store.chunk_counts().items(), key=lambda item: item[0][1])
        ]

    def _unused_path(self, name: str, doc_id: str) -> Path:
        path = self._data_dir / name
        if not path.exists():
            return path
        return self._data_dir / f"{name[:-4]}-{doc_id[:DISAMBIGUATING_ID_CHARS]}.pdf"
