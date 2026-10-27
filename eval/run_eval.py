"""Retrieval-only evaluation: measures search quality and the refusal threshold without any LLM calls.

Usage: python -m eval.run_eval [--k 4] [--min-similarity 0.25]
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from citewise.config import load_settings
from citewise.embedder import MiniLMEmbedder
from citewise.answer import Searcher
from citewise.indexing import index_directory
from citewise.reranker import CrossEncoderReranker
from citewise.retriever import HybridRetriever
from citewise.store import ChunkStore, Hit

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.jsonl"


@dataclass(frozen=True)
class Case:
    question: str
    page: int | None


@dataclass(frozen=True)
class Report:
    hit_at_1: float
    hit_at_k: float
    mrr: float
    refused_unanswerable: float
    answered_answerable: float


def load_cases(path: Path) -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def rank_of_page(hits: list[Hit], page: int) -> int | None:
    return next((rank for rank, hit in enumerate(hits, start=1) if hit.chunk.page == page), None)


def score(results: list[tuple[Case, list[Hit]]], min_similarity: float) -> Report:
    answerable = [(c, h) for c, h in results if c.page is not None]
    unanswerable = [h for c, h in results if c.page is None]
    ranks = [rank_of_page(h, c.page) for c, h in answerable if c.page is not None]

    def passes_floor(hits: list[Hit]) -> bool:
        # Mirrors Answerer: the gate uses the best meaning-search similarity, wherever the reranker placed it.
        return bool(hits) and max(h.similarity for h in hits) >= min_similarity

    return Report(
        hit_at_1=sum(r == 1 for r in ranks) / len(ranks),
        hit_at_k=sum(r is not None for r in ranks) / len(ranks),
        mrr=sum(1 / r for r in ranks if r is not None) / len(ranks),
        refused_unanswerable=sum(not passes_floor(h) for h in unanswerable) / max(len(unanswerable), 1),
        answered_answerable=sum(passes_floor(h) for _, h in answerable) / len(answerable),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    settings = load_settings()
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--min-similarity", type=float, default=settings.min_similarity)
    args = parser.parse_args()

    store = ChunkStore(None, "eval", MiniLMEmbedder(settings.embedding_model))
    index_directory(settings.data_dir, store, settings.max_chunk_chars)
    cases = load_cases(QUESTIONS_FILE)
    searchers: dict[str, Searcher] = {
        "dense": store,
        "hybrid": HybridRetriever(store, None, settings.candidate_pool),
        "hybrid+rerank": HybridRetriever(store, CrossEncoderReranker(settings.reranker_model), settings.candidate_pool),
    }

    reports: dict[str, Report] = {}
    for mode, searcher in searchers.items():
        results = [(case, searcher.search(case.question, args.k)) for case in cases]
        reports[mode] = score(results, args.min_similarity)
        misses = [c.question for c, h in results if c.page is not None and rank_of_page(h, c.page) != 1]
        print(f"{mode}: not ranked first -> {misses}")

    print(f"\nk={args.k} min_similarity={args.min_similarity}")
    print(f"{'metric':<22}" + "".join(f"{mode:>15}" for mode in reports))
    for metric in vars(next(iter(reports.values()))):
        print(f"{metric:<22}" + "".join(f"{getattr(r, metric):>15.2f}" for r in reports.values()))


if __name__ == "__main__":
    main()
