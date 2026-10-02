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
from rag_assistant.history import HistoryRepository, SQLiteHistoryRepository
from rag_assistant.indexing import Indexer
from rag_assistant.llm import LLMClient
from rag_assistant.llm.groq import GroqLLM
from rag_assistant.llm.ollama import OllamaLLM
from rag_assistant.reranking import Reranker
from rag_assistant.reranking.fastembed_reranker import FastEmbedReranker
from rag_assistant.reranking.noop import NoOpReranker
from rag_assistant.retrieval import Retriever
from rag_assistant.service.rag_service import RAGService
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


def _build_groq(settings: Settings) -> LLMClient:
    if settings.groq_api_key is None:
        raise ConfigurationError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
    return GroqLLM(
        api_key=settings.groq_api_key.get_secret_value(),
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )


_LLMS: dict[str, Callable[[Settings], LLMClient]] = {
    "ollama": lambda settings: OllamaLLM(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    ),
    "groq": _build_groq,
}


def build_llm(settings: Settings) -> LLMClient:
    builder = _LLMS.get(settings.llm_provider)
    if builder is None:
        raise ConfigurationError(
            f"Unknown LLM_PROVIDER={settings.llm_provider!r}; choose one of {sorted(_LLMS)}"
        )
    return builder(settings)


def build_history(settings: Settings) -> HistoryRepository:
    return SQLiteHistoryRepository(settings.history_db_path)


def build_rag_service(settings: Settings) -> RAGService:
    return RAGService(
        retriever=build_retriever(settings),
        llm=build_llm(settings),
        history=build_history(settings),
        history_max_messages=settings.history_max_messages,
        min_rerank_score=settings.min_rerank_score,
    )
