"""Test doubles for the Strategy interfaces, so unit tests never download a model."""

import hashlib

from rag_assistant.embeddings import Embedder


class FakeEmbedder(Embedder):
    """Deterministic embedder: each word lights up one of `dimension` buckets."""

    def __init__(self, dimension: int = 16) -> None:
        self._dimension = dimension
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for word in text.lower().split():
            bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self._dimension
            vector[bucket] += 1.0
        return vector if any(vector) else [1.0] + [0.0] * (self._dimension - 1)
