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
from rag_assistant.reranking import Reranker
from rag_assistant.reranking.fastembed_reranker import FastEmbedReranker
from rag_assistant.reranking.noop import NoOpReranker
from rag_assistant.retrieval import Retriever
from rag_assistant.vectorstore import QdrantVectorStore, VectorStore

_EMBEDDERS: dict[str, Callable[[Settings], Embedder]] = {
    "fastembed": lambda settings: FastEmbedEmbedder(
        settings.embedding_model,
        cache_dir=settings.model_cache_dir,
        batch_size=settings.embedding_batch_size,
    ),
}


def build_embedder(settings: Settings) -> Embedder:
    builder = _EMBEDDERS.get(settings.embedder_provider)
    if builder is None:
        raise ConfigurationError(
            f"Unknown EMBEDDER_PROVIDER={settings.embedder_provider!r}; "
            f"choose one of {sorted(_EMBEDDERS)}"
        )
    return builder(settings)


def build_splitter(settings: Settings) -> RecursiveTextSplitter:
    return RecursiveTextSplitter(settings.chunk_size, settings.chunk_overlap)


def build_vector_store(settings: Settings) -> VectorStore:
    client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
    return QdrantVectorStore(client, settings.qdrant_collection)


def build_indexer(settings: Settings) -> Indexer:
    return Indexer(
        build_splitter(settings),
        build_embedder(settings),
        build_vector_store(settings),
        batch_size=settings.embedding_batch_size,
    )


_RERANKERS: dict[str, Callable[[Settings], Reranker]] = {
    "fastembed": lambda settings: FastEmbedReranker(
        settings.reranker_model, cache_dir=settings.model_cache_dir
    ),
    "none": lambda settings: NoOpReranker(),
}


def build_reranker(settings: Settings) -> Reranker:
    builder = _RERANKERS.get(settings.reranker_provider)
    if builder is None:
        raise ConfigurationError(
            f"Unknown RERANKER_PROVIDER={settings.reranker_provider!r}; "
            f"choose one of {sorted(_RERANKERS)}"
        )
    return builder(settings)


def build_retriever(settings: Settings) -> Retriever:
    return Retriever(
        build_embedder(settings),
        build_vector_store(settings),
        build_reranker(settings),
        top_k=settings.top_k,
        rerank_top_n=settings.rerank_top_n,
    )
