"""RAGService (Facade pattern).

One method, ask(), hides the whole pipeline (history, question condensing,
retrieval, reranking, generation, persistence) from the API and the UI.
"""

import logging
import re
import time
from datetime import UTC, datetime

from rag_assistant.exceptions import HistoryError, LLMError
from rag_assistant.history import HistoryRepository
from rag_assistant.llm import LLMClient
from rag_assistant.models import ChatResult, RetrievedChunk, StoredMessage
from rag_assistant.retrieval import Retriever
from rag_assistant.service import prompts

logger = logging.getLogger(__name__)

# A rewritten question is one short line; anything longer is the model rambling.
_CONDENSE_MAX_TOKENS = 96
_CONDENSE_MAX_CHARS = 300
_CONDENSE_LABEL = re.compile(r"^(pregunta(\s+\w+)?|reescritura)\s*:\s*", re.IGNORECASE)


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
        history = self._load_history(session_id)

        standalone = self._condense(question, history)
        chunks = self._retriever.retrieve(standalone)

        if self._is_relevant(chunks):
            reply = self._llm.generate(prompts.answer_messages(question, chunks, history))
            answered = not prompts.is_no_answer(reply)
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
        self._save(result, question, asked_at, chunks)
        logger.info(
            "session=%s answered=%s latency_ms=%.0f sources=%d",
            session_id,
            answered,
            result.latency_ms,
            len(result.sources),
        )
        return result

    def _load_history(self, session_id: str) -> list[StoredMessage]:
        """The last N messages of the session that are worth showing to the LLM.

        Exchanges that ended in "no answer found" are left out: a small model
        that sees its own refusals tends to copy that sentence instead of
        answering or emitting the marker. A history failure degrades to "no
        memory" rather than blocking the answer.
        """
        try:
            messages = self._history.recent(session_id, self._history_max_messages)
        except HistoryError as exc:
            logger.warning("Could not load history for session=%s: %s", session_id, exc)
            return []

        usable: list[StoredMessage] = []
        for message in messages:
            if message.role == "assistant" and message.answered is False:
                if usable and usable[-1].role == "user":
                    usable.pop()
                continue
            usable.append(message)
        # An odd window can start with an answer whose question fell outside it.
        while usable and usable[0].role == "assistant":
            usable.pop(0)
        return usable

    def _condense(self, question: str, history: list[StoredMessage]) -> str:
        """Rewrite a follow-up ("¿y cuánto cuesta?") into a question that stands alone.

        Retrieval only sees this text, so without it follow-ups would search for
        pronouns. The model's output is not trusted: only its first line is
        kept, and if the LLM fails or rambles the original question is used. A
        weaker search is better than no answer.
        """
        if not history:
            return question
        try:
            reply = self._llm.generate(
                prompts.condense_messages(question, history), max_tokens=_CONDENSE_MAX_TOKENS
            )
        except LLMError as exc:
            logger.warning("Could not condense the question, using it as written: %s", exc)
            return question

        first_line = next((line.strip() for line in reply.splitlines() if line.strip()), "")
        rewritten = _CONDENSE_LABEL.sub("", first_line).strip(" \"'«»“”")
        if not rewritten or len(rewritten) > _CONDENSE_MAX_CHARS:
            logger.warning("Discarding unusable condensed question: %r", reply[:120])
            return question
        return rewritten

    def _is_relevant(self, chunks: list[RetrievedChunk]) -> bool:
        if not chunks:
            return False
        best = chunks[0].rerank_score
        # Without a reranker there is no calibrated score: let the LLM decide.
        return best is None or best >= self._min_rerank_score

    def _save(
        self, result: ChatResult, question: str, asked_at: datetime, chunks: list[RetrievedChunk]
    ) -> None:
        exchange = [
            StoredMessage(
                session_id=result.session_id, role="user", content=question, timestamp=asked_at
            ),
            StoredMessage(
                session_id=result.session_id,
                role="assistant",
                content=result.answer,
                timestamp=datetime.now(UTC),
                latency_ms=result.latency_ms,
                retrieved_urls=_unique_urls(chunks),
                rerank_scores=result.rerank_scores,
                answered=result.answered,
            ),
        ]
        try:
            self._history.add(exchange)
        except HistoryError as exc:
            # The answer is already computed: losing the log is better than losing the reply.
            logger.error("Could not save the exchange for session=%s: %s", result.session_id, exc)


def _unique_urls(chunks: list[RetrievedChunk]) -> list[str]:
    return list(dict.fromkeys(item.chunk.url for item in chunks))
