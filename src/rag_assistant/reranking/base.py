"""Reranker interface (Strategy pattern)."""

from abc import ABC, abstractmethod

from rag_assistant.models import RetrievedChunk


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """Return the `top_n` candidates most relevant to `query`, best first."""
