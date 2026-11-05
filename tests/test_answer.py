import pytest

from citewise.answer import NOT_FOUND, Answerer, build_prompt, cited_numbers

from tests.conftest import FakeLLM, FakeSearcher, make_hit


def make_answerer(hits, reply="SQL joins tables [1].", min_similarity=0.3):
    searcher = FakeSearcher(hits=hits)
    llm = FakeLLM(reply=reply)
    return Answerer(searcher=searcher, llm=llm, top_k=4, min_similarity=min_similarity), searcher, llm


class TestBuildPrompt:
    def test_numbers_sources_with_document_and_page(self):
        prompt = build_prompt("What is SQL?", [make_hit("SQL queries data.", page=7)])

        assert "[1] (notes.pdf, p. 7) SQL queries data." in prompt
        assert "Question: What is SQL?" in prompt

    def test_tells_the_model_how_to_refuse(self):
        prompt = build_prompt("q", [make_hit("text")])

        assert NOT_FOUND in prompt


class TestCitedNumbers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Fact [2]. Other [1][2].", [1, 2]),
            ("No citations here.", []),
            ("Out of range [9] and [0].", []),
        ],
    )
    def test_extracts_valid_unique_source_numbers(self, text, expected):
        assert cited_numbers(text, source_count=3) == expected


class TestAnswerer:
    def test_returns_answer_with_citations_for_cited_sources(self):
        hits = [make_hit("SQL joins tables.", page=7), make_hit("Charts show data.", page=5)]
        answerer, _, _ = make_answerer(hits, reply="SQL joins tables [1].")

        answer = answerer.ask("How do I join tables?")

        assert answer.found is True
        assert answer.text == "SQL joins tables [1]."
        assert [(c.number, c.page) for c in answer.citations] == [(1, 7)]

    def test_refuses_without_calling_llm_when_nothing_is_similar_enough(self):
        answerer, _, llm = make_answerer([make_hit("Unrelated.", similarity=0.1)])

        answer = answerer.ask("How do I bake bread?")

        assert answer.found is False
        assert answer.text == NOT_FOUND
        assert llm.prompts == []

    def test_keeps_keyword_only_hits_when_best_hit_clears_the_floor(self):
        # Keyword-only matches carry similarity 0.0; the floor is a question-level gate, not a per-hit filter.
        hits = [make_hit("Relevant.", similarity=0.9), make_hit("HAVING filters groups.", page=7, similarity=0.0)]
        answerer, _, llm = make_answerer(hits)

        answerer.ask("question")

        assert "Relevant." in llm.prompts[0]
        assert "HAVING filters groups." in llm.prompts[0]

    def test_refuses_when_best_hit_is_below_floor_even_if_reranked_first(self):
        hits = [make_hit("Reranked first.", similarity=0.1), make_hit("Also weak.", page=2, similarity=0.2)]
        answerer, _, llm = make_answerer(hits, min_similarity=0.3)

        assert answerer.ask("question").found is False
        assert llm.prompts == []

    def test_marks_answer_not_found_when_model_refuses(self):
        answerer, _, _ = make_answerer([make_hit("Some text.")], reply=NOT_FOUND)

        answer = answerer.ask("question")

        assert answer.found is False
        assert answer.citations == []

    def test_strips_whitespace_from_question_before_searching(self):
        answerer, searcher, _ = make_answerer([make_hit("x")])

        answerer.ask("  padded question  ")

        assert searcher.queries == ["padded question"]

    def test_snippet_is_trimmed_for_long_chunks(self):
        answerer, _, _ = make_answerer([make_hit("word " * 200)], reply="Answer [1].")

        citation = answerer.ask("q").citations[0]

        assert len(citation.snippet) <= 241
        assert citation.snippet.endswith("…")
