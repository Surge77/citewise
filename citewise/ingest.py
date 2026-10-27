import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

DOC_ID_LENGTH = 16


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    name: str
    pages: tuple[Page, ...]


def document_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:DOC_ID_LENGTH]


def load_pdf(path: Path) -> Document:
    data = path.read_bytes()
    reader = PdfReader(io.BytesIO(data))
    pages = tuple(
        Page(number=number, text=" ".join((page.extract_text() or "").split()))
        for number, page in enumerate(reader.pages, start=1)
    )
    return Document(doc_id=document_id(data), name=path.name, pages=pages)
