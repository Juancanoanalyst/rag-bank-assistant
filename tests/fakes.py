"""Test doubles for the Strategy interfaces, so unit tests never download a model."""

import hashlib

from rag_assistant.embeddings import Embedder
from rag_assistant.exceptions import LLMError
from rag_assistant.llm import LLMClient
from rag_assistant.models import ChatMessage, RetrievedChunk
from rag_assistant.reranking import Reranker


class FakeLLM(LLMClient):
    """Returns scripted replies in order and records every prompt it receives.

    A reply that is an exception instance is raised instead of returned.
    """

    def __init__(self, *replies: str | Exception) -> None:
        self._replies = list(replies)
        self.calls: list[list[ChatMessage]] = []
        self.max_tokens: list[int | None] = []

    def generate(self, messages: list[ChatMessage], max_tokens: int | None = None) -> str:
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
        if not self._replies:
            raise LLMError("FakeLLM has no scripted reply left")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeReranker(Reranker):
    """Scores a chunk by the share of query words it contains."""

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        words = set(query.lower().split())
        scored = [
            item.model_copy(
                update={
                    "rerank_score": len(words & set(item.chunk.text.lower().split())) / len(words)
                }
            )
            for item in candidates
        ]
        scored.sort(key=lambda item: item.rerank_score, reverse=True)
        return scored[:top_n]


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
