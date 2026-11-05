from collections.abc import Iterator
from dataclasses import dataclass

from citewise.answer import Answer, Answerer, Delta, Done
from citewise.config import get_api_key, load_settings
from citewise.embedder import MiniLMEmbedder
from citewise.indexing import index_directory
from citewise.library import DocumentInfo, Library, UploadResult
from citewise.llm import GeminiLLM
from citewise.reranker import CrossEncoderReranker
from citewise.retriever import HybridRetriever
from citewise.store import ChunkStore


@dataclass(frozen=True)
class CitewiseService:
    answerer: Answerer
    library: Library

    def ask(self, question: str) -> Answer:
        return self.answerer.ask(question)

    def ask_stream(self, question: str) -> Iterator[Delta | Done]:
        return self.answerer.ask_stream(question)

    def add_pdf(self, name: str, data: bytes) -> UploadResult:
        return self.library.add_pdf(name, data)

    def documents(self) -> list[DocumentInfo]:
        return self.library.documents()


def build_service() -> CitewiseService:
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = ChunkStore(settings.db_dir, settings.collection, MiniLMEmbedder(settings.embedding_model))
    index_directory(settings.data_dir, store, settings.max_chunk_chars)
    retriever = HybridRetriever(store, CrossEncoderReranker(settings.reranker_model), settings.candidate_pool)
    llm = GeminiLLM(api_key=get_api_key(), model=settings.llm_model, thinking_budget=settings.llm_thinking_budget)
    answerer = Answerer(searcher=retriever, llm=llm, top_k=settings.top_k, min_similarity=settings.min_similarity)
    return CitewiseService(answerer=answerer, library=Library(store, settings.data_dir, settings.max_chunk_chars))
