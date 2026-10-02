"""Qdrant-backed vector store."""

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_assistant.exceptions import IndexingError
from rag_assistant.models import Chunk

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """One Qdrant collection holding chunk vectors and their metadata."""

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def recreate(self, dimension: int) -> None:
        """Drop and create the collection, so no chunk of a previous run survives."""
        try:
            if self._client.collection_exists(self._collection):
                self._client.delete_collection(self._collection)
            self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        except Exception as exc:
            raise IndexingError(f"Could not create collection {self._collection!r}: {exc}") from exc

    def count(self) -> int:
        """Number of stored chunks; 0 when the collection does not exist yet."""
        try:
            if not self._client.collection_exists(self._collection):
                return 0
            return self._client.count(self._collection, exact=True).count
        except Exception as exc:
            raise IndexingError(f"Could not reach the vector store: {exc}") from exc

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise IndexingError(f"Got {len(chunks)} chunks but {len(vectors)} vectors")
        points = [
            PointStruct(id=chunk.id, vector=vector, payload=chunk.model_dump(exclude={"id"}))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            self._client.upsert(self._collection, points=points, wait=True)
        except Exception as exc:
            raise IndexingError(f"Could not write {len(points)} chunks: {exc}") from exc
