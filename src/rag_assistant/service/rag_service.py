"""RAGService (Facade pattern).

One method, ask(), hides the whole pipeline (history, question condensing,
retrieval, reranking, generation, persistence) from the API and the UI.
"""

import logging
import time
from datetime import UTC, datetime

from rag_assistant.exceptions import LLMError
from rag_assistant.history import HistoryRepository
from rag_assistant.llm import LLMClient
from rag_assistant.models import ChatResult, RetrievedChunk, StoredMessage
from rag_assistant.retrieval import Retriever
from rag_assistant.service import prompts

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        history: HistoryRepository,
        history_max_messages: int,
        min_rerank_score: float,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._history = history
        self._history_max_messages = history_max_messages
        self._min_rerank_score = min_rerank_score

    def ask(self, session_id: str, question: str) -> ChatResult:
        """Answer `question` within the conversation `session_id` and persist the exchange."""
        started = time.perf_counter()
        asked_at = datetime.now(UTC)
        history = self._history.recent(session_id, self._history_max_messages)

        standalone = self._condense(question, history)
        chunks = self._retriever.retrieve(standalone)

        if self._is_relevant(chunks):
            reply = self._llm.generate(prompts.answer_messages(question, chunks, history))
            answered = prompts.NO_ANSWER_MARKER not in reply.upper()
            answer = reply if answered else prompts.NO_ANSWER_MESSAGE
        else:
            answered, answer = False, prompts.NO_ANSWER_MESSAGE

        result = ChatResult(
            session_id=session_id,
            answer=answer,
            answered=answered,
            # Sources are only shown for real answers; the URLs are still stored for analytics.
            sources=_unique_urls(chunks) if answered else [],
            rerank_scores=[item.rerank_score for item in chunks if item.rerank_score is not None],
            standalone_question=standalone,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self._history.add(
            [
                StoredMessage(
                    session_id=session_id, role="user", content=question, timestamp=asked_at
                ),
                StoredMessage(
                    session_id=session_id,
                    role="assistant",
                    content=answer,
                    timestamp=datetime.now(UTC),
                    latency_ms=result.latency_ms,
                    retrieved_urls=_unique_urls(chunks),
                    rerank_scores=result.rerank_scores,
                    answered=answered,
                ),
            ]
        )
        logger.info(
            "session=%s answered=%s latency_ms=%.0f sources=%d",
            session_id,
            answered,
            result.latency_ms,
            len(result.sources),
        )
        return result

    def _condense(self, question: str, history: list[StoredMessage]) -> str:
        """Rewrite a follow-up ("¿y cuánto cuesta?") into a question that stands alone.

        Retrieval only sees this text, so without it follow-ups would search for
        pronouns. If the LLM fails here the original question is used: a weaker
        search is better than no answer.
        """
        if not history:
            return question
        try:
            return self._llm.generate(prompts.condense_messages(question, history))
        except LLMError as exc:
            logger.warning("Could not condense the question, using it as written: %s", exc)
            return question

    def _is_relevant(self, chunks: list[RetrievedChunk]) -> bool:
        if not chunks:
            return False
        best = chunks[0].rerank_score
        # Without a reranker there is no calibrated score: let the LLM decide.
        return best is None or best >= self._min_rerank_score


def _unique_urls(chunks: list[RetrievedChunk]) -> list[str]:
    return list(dict.fromkeys(item.chunk.url for item in chunks))
