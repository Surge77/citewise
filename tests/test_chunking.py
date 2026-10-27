from citewise.chunking import chunk_document, split_sentences

from tests.conftest import make_document


class TestSplitSentences:
    def test_splits_on_sentence_punctuation(self):
        assert split_sentences("One fact. Two facts! Three?") == ["One fact.", "Two facts!", "Three?"]

    def test_splits_sentences_glued_without_a_space(self):
        assert split_sentences("Ends here.Next starts") == ["Ends here.", "Next starts"]

    def test_returns_empty_list_for_blank_text(self):
        assert split_sentences("   ") == []


class TestChunkDocument:
    def test_every_chunk_keeps_its_page_number(self):
        doc = make_document("Page one fact.", "Page two fact.")

        chunks = chunk_document(doc, max_chars=500)

        assert [(c.page, c.text) for c in chunks] == [(1, "Page one fact."), (2, "Page two fact.")]

    def test_chunks_never_span_two_pages(self):
        doc = make_document("Alpha fact.", "Beta fact.")

        chunks = chunk_document(doc, max_chars=10_000)

        assert len(chunks) == 2

    def test_drops_sentences_repeated_anywhere_in_the_document(self):
        doc = make_document("Data is useful. Data is useful. More here.", "data is   USEFUL. Other.")

        texts = [c.text for c in chunk_document(doc, max_chars=500)]

        assert texts == ["Data is useful. More here.", "Other."]

    def test_drops_repeat_that_lost_its_heading_prefix(self):
        doc = make_document("Page 1: Intro Data is useful.Data is useful.")

        texts = [c.text for c in chunk_document(doc, max_chars=500)]

        assert texts == ["Page 1: Intro Data is useful."]

    def test_packs_sentences_up_to_max_chars_with_one_sentence_overlap(self):
        doc = make_document("Aaaa aaaa. Bbbb bbbb. Cccc cccc.")

        texts = [c.text for c in chunk_document(doc, max_chars=22)]

        assert texts == ["Aaaa aaaa. Bbbb bbbb.", "Bbbb bbbb. Cccc cccc."]

    def test_oversized_sentence_becomes_its_own_chunk(self):
        long_sentence = "Word " * 50 + "end."
        doc = make_document(f"Short one. {long_sentence}")

        texts = [c.text for c in chunk_document(doc, max_chars=40)]

        assert texts[-1] == long_sentence.strip()

    def test_chunk_ids_are_deterministic_and_unique(self):
        doc = make_document("A fact. B fact.", "C fact.", doc_id="abc")

        ids = [c.chunk_id for c in chunk_document(doc, max_chars=8)]

        assert ids == ["abc:p1:c0", "abc:p1:c1", "abc:p2:c0"]

    def test_skips_pages_without_text(self):
        doc = make_document("", "Real text.")

        chunks = chunk_document(doc, max_chars=500)

        assert [c.page for c in chunks] == [2]
