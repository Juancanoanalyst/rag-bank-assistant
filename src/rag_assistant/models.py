"""Data shapes passed between pipeline stages."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """One turn sent to an LLM."""

    role: Literal["system", "user", "assistant"]
    content: str


class StoredMessage(BaseModel):
    """A row of the conversation history.

    The metric fields are only set on assistant messages: they describe how
    that answer was produced.
    """

    session_id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    latency_ms: float | None = None
    retrieved_urls: list[str] = []
    rerank_scores: list[float] = []
    # False when the assistant replied "no answer found"; None on user messages.
    answered: bool | None = None


class ChatResult(BaseModel):
    """What RAGService returns for one question."""

    session_id: str
    answer: str
    answered: bool
    sources: list[str]
    rerank_scores: list[float]
    # The question as rewritten for retrieval using the conversation so far.
    standalone_question: str
    latency_ms: float


class RawPage(BaseModel):
    """An HTML page as stored in data/raw, before any cleaning."""

    url: str
    html: str
    filename: str


class Document(BaseModel):
    """A cleaned page: one line of data/clean/documents.jsonl."""

    url: str
    title: str
    section: str
    text: str


class Chunk(BaseModel):
    """A slice of a Document: the unit that is embedded, stored and retrieved."""

    id: str
    url: str
    title: str
    section: str
    chunk_index: int
    text: str

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedder: the page title gives a short chunk its context."""
        return f"{self.title}\n{self.text}"


class RetrievedChunk(BaseModel):
    """A chunk returned for a query, with the scores that ranked it."""

    chunk: Chunk
    # Cosine similarity from the vector search (first stage).
    vector_score: float
    # Relevance in [0, 1] from the cross-encoder; None when reranking is disabled.
    rerank_score: float | None = None
