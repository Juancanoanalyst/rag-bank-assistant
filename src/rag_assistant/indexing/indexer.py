"""Ingestion stage 3: chunk clean documents, embed them and load the vector store."""

import logging
from dataclasses import dataclass
from itertools import batched

from rag_assistant.chunking import RecursiveTextSplitter
from rag_assistant.embeddings import Embedder
from rag_assistant.exceptions import IndexingError
from rag_assistant.models import Document
from rag_assistant.vectorstore import QdrantVectorStore

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
        store: QdrantVectorStore,
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

        With `skip_if_indexed`, an already populated collection is left alone,
        which makes container start-up idempotent.
        """
        if skip_if_indexed and (existing := self._store.count()):
            logger.info("Collection already holds %d chunks; skipping indexing", existing)
            return IndexReport(skipped=True)
        if not documents:
            raise IndexingError("There are no documents to index; run the scraping step first")

        chunks = list(self._splitter.split_documents(documents))
        self._store.recreate(self._embedder.dimension)
        for number, batch in enumerate(batched(chunks, self._batch_size), start=1):
            vectors = self._embedder.embed_documents([chunk.embedding_text for chunk in batch])
            self._store.upsert(list(batch), vectors)
            logger.info(
                "Indexed %d/%d chunks", min(number * self._batch_size, len(chunks)), len(chunks)
            )

        logger.info("Indexed %d documents as %d chunks", len(documents), len(chunks))
        return IndexReport(documents=len(documents), chunks=len(chunks))
