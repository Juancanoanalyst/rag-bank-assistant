"""Embedder interface (Strategy pattern).

The indexer and the retriever depend only on this interface, so the embedding
backend can be swapped through configuration without touching either of them.
"""

from abc import ABC, abstractmethod


class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Size of the vectors this embedder produces."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages to be stored in the vector database."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a user question for similarity search."""
