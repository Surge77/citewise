from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MiniLMEmbedder:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: "SentenceTransformer | None" = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._model
        if model is None:
            # Imported lazily: sentence-transformers pulls in torch, which takes seconds and ~1 GB of RAM.
            from sentence_transformers import SentenceTransformer

            model = self._model = SentenceTransformer(self._model_name)
        return model.encode(texts, normalize_embeddings=True).tolist()
