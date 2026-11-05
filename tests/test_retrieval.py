from dataclasses import dataclass, field

from citewise.chunking import Chunk
from citewise.keyword import BM25Index, tokenize
from citewise.retriever import HybridRetriever, reciprocal_rank_fusion
from citewise.store import ChunkStore


def make_chunk(text: str, page: int) -> Chunk:
    return Chunk(chunk_id=f"doc1:p{page}:c0", doc_id="doc1", doc_name="notes.pdf", page=page, text=text)


@dataclass
class StubReranker:
    """Scores a text by how many times it contains the query's first word."""

    calls: list[list[str]] = field(default_factory=list)

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append(texts)
        first = query.lower().split()[0]
        return [float(text.lower().count(first)) for text in texts]


class TestTokenize:
    def test_lowercases_and_keeps_alphanumeric_words(self):
        assert tokenize("SQL's GROUP-BY, k8s!") == ["sql", "s", "group", "by", "k8s"]


class TestBM25Index:
    def test_ranks_chunk_with_rare_query_term_first(self):
        index = BM25Index([make_chunk("data is useful data", 1), make_chunk("having filters groups data", 2)])

        results = index.search("having", k=2)

        assert [chunk.page for chunk, _ in results] == [2]

    def test_excludes_chunks_sharing_no_terms(self):
        index = BM25Index([make_chunk("charts and plots", 1)])

        assert index.search("sql join", k=5) == []

    def test_empty_index_returns_nothing(self):
        assert BM25Index([]).search("anything", k=3) == []


class TestReciprocalRankFusion:
    def test_item_ranked_well_in_both_lists_wins(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])

        assert fused[0] == "b"

    def test_keeps_items_found_by_only_one_list(self):
        fused = reciprocal_rank_fusion([["a"], ["z"]])

        assert set(fused) == {"a", "z"}


class TestHybridRetriever:
    def test_finds_keyword_only_match_that_dense_search_misses(self, tmp_path, keyword_embedder):
        store = ChunkStore(tmp_path / "db", "retrieval", keyword_embedder)
        # Pool of 1: meaning search returns a chart chunk; BM25 ranks "having" first because it is the rarer term.
        store.add([make_chunk("chart visualization", 1), make_chunk("having clause", 2), make_chunk("chart model", 3)])

        hits = HybridRetriever(store, reranker=None, candidates=1).search("having chart", k=2)

        keyword_hit = next(h for h in hits if h.chunk.page == 2)
        assert keyword_hit.similarity == 0.0

    def test_reranker_decides_final_order_and_result_is_cut_to_k(self, tmp_path, keyword_embedder):
        store = ChunkStore(tmp_path / "db", "retrieval", keyword_embedder)
        store.add([make_chunk("sql sql", 1), make_chunk("model model model", 2), make_chunk("chart", 3)])
        reranker = StubReranker()

        hits = HybridRetriever(store, reranker=reranker, candidates=10).search("model sql chart", k=1)

        assert [h.chunk.page for h in hits] == [2]
        assert len(reranker.calls) == 1

    def test_keyword_index_picks_up_chunks_added_later(self, tmp_path, keyword_embedder):
        store = ChunkStore(tmp_path / "db", "retrieval", keyword_embedder)
        store.add([make_chunk("chart", 1)])
        retriever = HybridRetriever(store, reranker=None, candidates=10)
        retriever.search("having", k=1)

        store.add([make_chunk("having clause", 2)])

        assert any(h.chunk.page == 2 for h in retriever.search("having", k=2))

    def test_empty_store_returns_nothing(self, tmp_path, keyword_embedder):
        store = ChunkStore(tmp_path / "db", "retrieval", keyword_embedder)

        assert HybridRetriever(store, reranker=None, candidates=10).search("q", k=3) == []
