"""Reranker used when RERANKER_PROVIDER=none: keeps the vector-search order."""

from rag_assistant.models import RetrievedChunk
from rag_assistant.reranking.base import Reranker


class NoOpReranker(Reranker):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return candidates[:top_n]
