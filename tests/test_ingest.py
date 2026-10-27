from pathlib import Path

import pytest

from citewise.ingest import document_id, load_pdf

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "data" / "DA_Notes.pdf"

# The 10-page sample notes PDF is not redistributed with the repo (unknown author); these tests run when it is present.
requires_sample_pdf = pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF data/DA_Notes.pdf not present")


def test_document_id_is_stable_for_identical_bytes():
    assert document_id(b"same bytes") == document_id(b"same bytes")
    assert document_id(b"same bytes") != document_id(b"other bytes")


@requires_sample_pdf
def test_load_pdf_returns_one_numbered_page_per_pdf_page():
    doc = load_pdf(SAMPLE_PDF)

    assert doc.name == "DA_Notes.pdf"
    assert [p.number for p in doc.pages] == list(range(1, 11))


@requires_sample_pdf
def test_load_pdf_collapses_whitespace():
    doc = load_pdf(SAMPLE_PDF)

    assert all("\n" not in p.text and "  " not in p.text for p in doc.pages)
