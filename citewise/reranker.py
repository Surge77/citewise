from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Reads the question and each passage together, which judges relevance better than comparing two vectors."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: "CrossEncoder | None" = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        model = self._model
        if model is None:
            # Lazy import for the same reason as the embedder: it loads torch.
            from sentence_transformers import CrossEncoder

            model = self._model = CrossEncoder(self._model_name)
        return model.predict([(query, text) for text in texts]).tolist()
