"""Two-stage retrieval: vector search for recall, cross-encoder for precision."""

import logging

from rag_assistant.embeddings import Embedder
from rag_assistant.models import Chunk, RetrievedChunk
from rag_assistant.reranking import Reranker
from rag_assistant.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker,
        top_k: int,
        rerank_top_n: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._top_k = top_k
        self._rerank_top_n = rerank_top_n

    def warm_up(self) -> None:
        """Force both models to load by running them once on a throwaway input."""
        sample = Chunk(id="warm-up", url="", title="", section="", chunk_index=0, text="hola")
        self._embedder.embed_query("hola")
        self._reranker.rerank("hola", [RetrievedChunk(chunk=sample, vector_score=0.0)], top_n=1)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Return the chunks most relevant to `query`, best first.

        TOP_K candidates come from the vector store; the reranker keeps the
        RERANK_TOP_N best. An empty list means nothing matched.
        """
        candidates = self._store.search(self._embedder.embed_query(query), self._top_k)
        results = self._reranker.rerank(query, candidates, self._rerank_top_n)
        logger.info("Retrieved %d candidates, kept %d for %r", len(candidates), len(results), query)
        return results
