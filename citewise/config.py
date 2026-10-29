import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_MODELS_ENV = "CITEWISE_LLM_MODELS"
# Tried in order; each model has its own free-tier quota, so the next one takes over when one is exhausted.
DEFAULT_LLM_MODELS = ("gemini-2.5-flash", "gemini-3.5-flash-lite")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = PROJECT_ROOT / "data"
    db_dir: Path = PROJECT_ROOT / ".citewise_db"
    collection: str = "citewise"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_models: tuple[str, ...] = DEFAULT_LLM_MODELS
    # 0 disables Gemini 2.5's hidden reasoning: measured first text ~1.0s instead of ~5.0s. Answers here are short
    # and grounded in retrieved sources, so the reasoning step adds latency without much benefit.
    llm_thinking_budget: int = 0
    top_k: int = 4
    # Libraries up to this size are sent to the model whole (~15k tokens), so summaries, lists, counts and
    # comparisons work; larger libraries fall back to search. Gemini Flash accepts far more, but cost grows per token.
    full_context_chars: int = 60_000
    # How many results each search (meaning and keyword) contributes before fusion and reranking.
    candidate_pool: int = 20
    # Below this cosine similarity a chunk is treated as unrelated. From eval/run_eval.py: off-topic questions
    # scored <= 0.06 and real ones >= 0.24, so 0.15 sits mid-gap. Re-tune when documents or the model change.
    min_similarity: float = 0.15
    max_chunk_chars: int = 600


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    models = tuple(m.strip() for m in os.environ.get(LLM_MODELS_ENV, "").split(",") if m.strip())
    return Settings(llm_models=models or DEFAULT_LLM_MODELS)


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY in your environment or .env file.")
    return key
