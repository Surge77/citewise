from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from citewise.chunking import Chunk
from citewise.embedder import Embedder


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    similarity: float


class ChunkStore:
    def __init__(self, db_dir: Path | None, collection: str, embedder: Embedder) -> None:
        """Persist to db_dir, or keep everything in memory when db_dir is None."""
        # In-memory mode exists because on Windows a persistent client holds its index files open until process exit.
        client = chromadb.EphemeralClient() if db_dir is None else chromadb.PersistentClient(path=str(db_dir))
        self._collection = client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        self._embedder = embedder

    def count(self) -> int:
        return self._collection.count()

    def has_document(self, doc_id: str) -> bool:
        return bool(self._collection.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def all_chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []
        return [_to_chunk(i, t, m) for i, t, m in zip(result["ids"], documents, metadatas)]

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=self._embedder.embed(texts),  # type: ignore[arg-type]  # chroma accepts plain float lists
            documents=texts,
            metadatas=[{"doc_id": c.doc_id, "doc_name": c.doc_name, "page": c.page} for c in chunks],
        )

    def search(self, query: str, k: int) -> list[Hit]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=self._embedder.embed([query]),  # type: ignore[arg-type]
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        # Cosine distance is 1 - cosine similarity.
        return [
            Hit(chunk=_to_chunk(chunk_id, text, meta), similarity=1.0 - distance)
            for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances)
        ]


def _to_chunk(chunk_id: str, text: str, meta: Mapping[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=str(meta["doc_id"]),
        doc_name=str(meta["doc_name"]),
        page=int(str(meta["page"])),
        text=text,
    )
