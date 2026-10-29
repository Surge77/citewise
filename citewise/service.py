from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from citewise.answer import Answer, Answerer, Delta, Done, Turn
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

    def ask(self, question: str, history: Sequence[Turn] = ()) -> Answer:
        return self.answerer.ask(question, history)

    def ask_stream(self, question: str, history: Sequence[Turn] = ()) -> Iterator[Delta | Done]:
        return self.answerer.ask_stream(question, history)

    def add_pdf(self, name: str, data: bytes) -> UploadResult:
        return self.library.add_pdf(name, data)

    def documents(self) -> list[DocumentInfo]:
        return self.library.documents()


def build_service() -> CitewiseService:
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    embedder = MiniLMEmbedder(settings.embedding_model)
    reranker = CrossEncoderReranker(settings.reranker_model)
    # Load both models now: a missing model then fails at startup with a clear error instead of breaking the
    # first question, and the first question doesn't pay the load time.
    embedder.embed(["warm-up"])
    reranker.score("warm-up", ["warm-up"])
    store = ChunkStore(settings.db_dir, settings.collection, embedder)
    index_directory(settings.data_dir, store, settings.max_chunk_chars)
    retriever = HybridRetriever(store, reranker, settings.candidate_pool)
    llm = GeminiLLM(
        api_key=get_api_key(), models=list(settings.llm_models), thinking_budget=settings.llm_thinking_budget
    )
    answerer = Answerer(
        searcher=retriever,
        llm=llm,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        library=store,
        full_context_chars=settings.full_context_chars,
    )
    return CitewiseService(answerer=answerer, library=Library(store, settings.data_dir, settings.max_chunk_chars))
