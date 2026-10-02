"""Domain exceptions.

Every layer raises a subclass of RAGError so the API can map failures to HTTP
responses without knowing about requests, qdrant or ollama internals.
"""


class RAGError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(RAGError):
    """Settings are missing or inconsistent."""


class ScrapingError(RAGError):
    """A page or sitemap could not be fetched or parsed."""


class IndexingError(RAGError):
    """Chunks could not be embedded or written to the vector store."""


class RetrievalError(RAGError):
    """The vector store or the reranker failed while answering a query."""


class LLMError(RAGError):
    """The LLM provider failed or returned an unusable response."""


class HistoryError(RAGError):
    """The conversation history store failed."""
