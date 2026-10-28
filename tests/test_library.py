import pytest

from citewise.library import InvalidPDFError, Library, safe_filename
from citewise.store import ChunkStore

from tests.pdf_factory import make_pdf_bytes


class TestSafeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("notes.pdf", "notes.pdf"),
            ("../../etc/passwd.pdf", "passwd.pdf"),
            ("C:\\Users\\someone\\report.pdf", "report.pdf"),
            ("rep<o>rt?.PDF", "rep_o_rt_.pdf"),
            ("no-extension", "no-extension.pdf"),
            (".hidden.pdf", "hidden.pdf"),
            ("", "document.pdf"),
            ("../", "document.pdf"),
        ],
    )
    def test_strips_paths_and_unsafe_characters(self, raw, expected):
        assert safe_filename(raw) == expected

    def test_truncates_very_long_names(self):
        assert len(safe_filename("a" * 500 + ".pdf")) <= 104


@pytest.fixture
def library(tmp_path, keyword_embedder):
    store = ChunkStore(tmp_path / "db", "library", keyword_embedder)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return Library(store=store, data_dir=data_dir, max_chunk_chars=500), data_dir


class TestLibrary:
    def test_indexes_new_pdf_and_saves_it_under_a_safe_name(self, library):
        lib, data_dir = library

        result = lib.add_pdf("../Sales Report.pdf", make_pdf_bytes("Sql joins tables.", "Charts show trends."))

        assert result.name == "Sales Report.pdf"
        assert (result.pages, result.chunks_added, result.already_indexed) == (2, 2, False)
        assert (data_dir / "Sales Report.pdf").exists()

    def test_uploading_identical_bytes_again_is_a_no_op(self, library):
        lib, data_dir = library
        pdf = make_pdf_bytes("Sql joins tables.")
        lib.add_pdf("a.pdf", pdf)

        again = lib.add_pdf("renamed.pdf", pdf)

        assert again.already_indexed is True
        assert again.chunks_added == 0
        assert sorted(p.name for p in data_dir.iterdir()) == ["a.pdf"]

    def test_same_name_with_different_content_gets_a_distinct_filename(self, library):
        lib, data_dir = library
        lib.add_pdf("notes.pdf", make_pdf_bytes("First version."))

        second = lib.add_pdf("notes.pdf", make_pdf_bytes("Second version."))

        assert second.name != "notes.pdf"
        assert second.name.startswith("notes-") and second.name.endswith(".pdf")
        assert len(list(data_dir.iterdir())) == 2

    def test_reports_pages_without_text(self, library):
        lib, _ = library

        result = lib.add_pdf("scan.pdf", make_pdf_bytes("", "Real text here."))

        assert result.empty_pages == [1]

    def test_rejects_files_that_are_not_pdfs(self, library):
        lib, data_dir = library

        with pytest.raises(InvalidPDFError, match="not a PDF"):
            lib.add_pdf("evil.pdf", b"MZ\x90\x00 this is an executable")

        assert list(data_dir.iterdir()) == []

    def test_rejects_corrupt_pdfs(self, library):
        lib, _ = library

        with pytest.raises(InvalidPDFError, match="could not be read"):
            lib.add_pdf("broken.pdf", b"%PDF-1.4\n garbage without structure")

    def test_lists_documents_with_chunk_counts(self, library):
        lib, _ = library
        lib.add_pdf("one.pdf", make_pdf_bytes("Sql joins.", "Charts help."))
        lib.add_pdf("two.pdf", make_pdf_bytes("Models predict."))

        docs = sorted((d.name, d.chunks) for d in lib.documents())

        assert docs == [("one.pdf", 2), ("two.pdf", 1)]
