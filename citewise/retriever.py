from collections import defaultdict

from citewise.keyword import BM25Index
from citewise.reranker import Reranker
from citewise.store import ChunkStore, Hit

# Damping constant from the original reciprocal rank fusion paper; stops a single #1 rank from dominating.
RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[str]]) -> list[str]:
    """Merge ranked id lists by rank alone, so BM25 scores and cosine similarities never need a common scale."""
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (RRF_K + rank)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


class HybridRetriever:
    def __init__(self, store: ChunkStore, reranker: Reranker | None, candidates: int) -> None:
        self._store = store
        self._reranker = reranker
        self._candidates = candidates
        self._keyword_index = BM25Index([])
        self._indexed_count = 0

    def _keyword_search(self, query: str) -> list[Hit]:
        count = self._store.count()
        if count != self._indexed_count:
            self._keyword_index = BM25Index(self._store.all_chunks())
            self._indexed_count = count
        # Similarity 0.0 marks "not found by meaning search"; the refusal gate only trusts dense similarity.
        return [Hit(chunk=chunk, similarity=0.0) for chunk, _ in self._keyword_index.search(query, self._candidates)]

    def search(self, query: str, k: int) -> list[Hit]:
        dense = self._store.search(query, self._candidates)
        keyword = self._keyword_search(query)
        by_id = {h.chunk.chunk_id: h for h in keyword} | {h.chunk.chunk_id: h for h in dense}
        fused_ids = reciprocal_rank_fusion([[h.chunk.chunk_id for h in dense], [h.chunk.chunk_id for h in keyword]])
        candidates = [by_id[chunk_id] for chunk_id in fused_ids]
        if self._reranker is not None and candidates:
            scores = self._reranker.score(query, [h.chunk.text for h in candidates])
            ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
            candidates = [hit for _, hit in ranked]
        return candidates[:k]
