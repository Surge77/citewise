from citewise.answer import Answerer
from citewise.config import get_api_key, load_settings
from citewise.embedder import MiniLMEmbedder
from citewise.indexing import index_directory
from citewise.llm import GeminiLLM
from citewise.reranker import CrossEncoderReranker
from citewise.retriever import HybridRetriever
from citewise.store import ChunkStore


def build_answerer() -> Answerer:
    settings = load_settings()
    store = ChunkStore(settings.db_dir, settings.collection, MiniLMEmbedder(settings.embedding_model))
    index_directory(settings.data_dir, store, settings.max_chunk_chars)
    retriever = HybridRetriever(store, CrossEncoderReranker(settings.reranker_model), settings.candidate_pool)
    llm = GeminiLLM(api_key=get_api_key(), model=settings.llm_model)
    return Answerer(searcher=retriever, llm=llm, top_k=settings.top_k, min_similarity=settings.min_similarity)
