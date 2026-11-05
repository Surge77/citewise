import math
import re
from collections import Counter

from citewise.chunking import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
# Standard Okapi BM25 parameters: k1 caps how much repeated terms help, b sets the document-length penalty.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Index:
    """Exact-term search; complements embeddings, which blur rare terms like "HAVING" or "DAX"."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._term_counts = [Counter(tokenize(c.text)) for c in chunks]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._avg_length = sum(self._lengths) / len(chunks) if chunks else 0.0
        doc_freq = Counter(term for counts in self._term_counts for term in counts)
        total = len(chunks)
        self._idf = {term: math.log(1 + (total - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def _score(self, terms: list[str], counts: Counter[str], length: int) -> float:
        norm = K1 * (1 - B + B * length / self._avg_length)
        return sum(self._idf[t] * counts[t] * (K1 + 1) / (counts[t] + norm) for t in terms if t in counts)

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored = [
            (chunk, self._score(terms, counts, length))
            for chunk, counts, length in zip(self._chunks, self._term_counts, self._lengths)
        ]
        matches = [(chunk, score) for chunk, score in scored if score > 0]
        return sorted(matches, key=lambda pair: pair[1], reverse=True)[:k]
