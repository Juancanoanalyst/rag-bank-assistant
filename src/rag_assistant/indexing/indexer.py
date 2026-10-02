"""Ingestion stage 3: chunk clean documents, embed them and load the vector store."""

import logging
from dataclasses import dataclass
from itertools import batched

from rag_assistant.chunking import RecursiveTextSplitter
from rag_assistant.embeddings import Embedder
from rag_assistant.exceptions import IndexingError
from rag_assistant.models import Document
from rag_assistant.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    documents: int = 0
    chunks: int = 0
    skipped: bool = False


class Indexer:
    def __init__(
        self,
        splitter: RecursiveTextSplitter,
        embedder: Embedder,
        store: VectorStore,
        batch_size: int = 32,
    ) -> None:
        self._splitter = splitter
        self._embedder = embedder
        self._store = store
        self._batch_size = batch_size

    def index(self, documents: list[Document], skip_if_indexed: bool = False) -> IndexReport:
        """Rebuild the collection from `documents`.

        The collection is recreated on every run: it is small, and starting
        clean guarantees that chunks of pages removed from the site, or left
        over from a different chunk size, never linger.

        Everything is chunked and embedded before the existing collection is
        touched, so a failure while embedding keeps the previous index. If
        writing fails, the half-built collection is dropped rather than left
        looking complete.

        With `skip_if_indexed`, a populated collection whose vector size matches
        the current embedder is left alone, which makes container start-up
        idempotent.
        """
        if skip_if_indexed and self._is_already_indexed():
            return IndexReport(skipped=True)

        chunks = list(self._splitter.split_documents(documents))
        if not chunks:
            raise IndexingError("There is no text to index; run the scraping step first")

        vectors: list[list[float]] = []
        for batch in batched(chunks, self._batch_size):
            vectors.extend(
                self._embedder.embed_documents([chunk.embedding_text for chunk in batch])
            )
            logger.info("Embedded %d/%d chunks", len(vectors), len(chunks))

        self._store.recreate(self._embedder.dimension)
        try:
            for start in range(0, len(chunks), self._batch_size):
                end = start + self._batch_size
                self._store.upsert(chunks[start:end], vectors[start:end])
        except IndexingError:
            self._store.drop()
            raise

        logger.info("Indexed %d documents as %d chunks", len(documents), len(chunks))
        return IndexReport(documents=len(documents), chunks=len(chunks))

    def _is_already_indexed(self) -> bool:
        existing = self._store.count()
        if not existing:
            return False
        stored_size = self._store.vector_size()
        if stored_size != self._embedder.dimension:
            logger.warning(
                "Collection holds vectors of size %s but the embedder produces %d; re-indexing",
                stored_size,
                self._embedder.dimension,
            )
            return False
        logger.info("Collection already holds %d chunks; skipping indexing", existing)
        return True
