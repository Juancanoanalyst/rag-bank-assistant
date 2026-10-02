"""Factories (Factory pattern): build components from Settings.

This is the only module that knows which concrete class sits behind each
interface. Everything else receives ready-made objects, so choosing another
embedder or vector store is a configuration change, not a code change.
"""

from collections.abc import Callable

from qdrant_client import QdrantClient

from rag_assistant.chunking import RecursiveTextSplitter
from rag_assistant.config import Settings
from rag_assistant.embeddings import Embedder
from rag_assistant.embeddings.fastembed_embedder import FastEmbedEmbedder
from rag_assistant.exceptions import ConfigurationError
from rag_assistant.indexing import Indexer
from rag_assistant.vectorstore import QdrantVectorStore

_EMBEDDERS: dict[str, Callable[[Settings], Embedder]] = {
    "fastembed": lambda settings: FastEmbedEmbedder(
        settings.embedding_model,
        cache_dir=settings.model_cache_dir,
        batch_size=settings.embedding_batch_size,
    ),
}


def build_embedder(settings: Settings) -> Embedder:
    try:
        return _EMBEDDERS[settings.embedder_provider](settings)
    except KeyError:
        raise ConfigurationError(
            f"Unknown EMBEDDER_PROVIDER={settings.embedder_provider!r}; "
            f"choose one of {sorted(_EMBEDDERS)}"
        ) from None


def build_splitter(settings: Settings) -> RecursiveTextSplitter:
    return RecursiveTextSplitter(settings.chunk_size, settings.chunk_overlap)


def build_vector_store(settings: Settings) -> QdrantVectorStore:
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
    return QdrantVectorStore(client, settings.qdrant_collection)


def build_indexer(settings: Settings) -> Indexer:
    return Indexer(
        build_splitter(settings),
        build_embedder(settings),
        build_vector_store(settings),
        batch_size=settings.embedding_batch_size,
    )
