"""Vector store interface.

The indexer and the retriever depend on this abstraction rather than on the
Qdrant client, which keeps Qdrant-specific code in one module.
"""

from abc import ABC, abstractmethod

from rag_assistant.models import Chunk


class VectorStore(ABC):
    @abstractmethod
    def recreate(self, dimension: int) -> None:
        """Create an empty collection for vectors of `dimension`, replacing any existing one."""

    @abstractmethod
    def drop(self) -> None:
        """Delete the collection if it exists."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored chunks; 0 when the collection does not exist."""

    @abstractmethod
    def vector_size(self) -> int | None:
        """Vector dimension of the collection; None when it does not exist."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store chunks with their vectors, overwriting chunks that share an id."""
