"""Usage and impact metrics computed from the conversation history."""

from collections import Counter

from pydantic import BaseModel

from rag_assistant.models import StoredMessage

TOP_URLS = 10


class UrlCount(BaseModel):
    url: str
    count: int


class SessionSummary(BaseModel):
    session_id: str
    messages: int
    questions: int


class Metrics(BaseModel):
    sessions: int
    messages: int
    questions: int
    answered: int
    avg_messages_per_session: float
    # Share of questions that ended in "no answer found" (0-1).
    no_answer_rate: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    # Mean of the best rerank score of each question: how well the corpus covers what is asked.
    avg_top_rerank_score: float | None
    # Answered questions x the configured minutes a manual search would have taken.
    estimated_hours_saved: float
    manual_search_minutes: float
    top_urls: list[UrlCount]
    messages_per_session: list[SessionSummary]


def percentile(values: list[float], fraction: float) -> float | None:
    """Percentile with linear interpolation between the two nearest values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compute_metrics(messages: list[StoredMessage], manual_search_minutes: float) -> Metrics:
    """Summarise the whole history.

    Each assistant message is one question: failed requests are not stored, so
    questions and answers are always written as a pair.
    """
    answers = [message for message in messages if message.role == "assistant"]
    answered = sum(1 for message in answers if message.answered)
    latencies = [message.latency_ms for message in answers if message.latency_ms is not None]
    top_scores = [max(message.rerank_scores) for message in answers if message.rerank_scores]
    url_counts = Counter(url for message in answers for url in message.retrieved_urls)

    per_session: dict[str, SessionSummary] = {}
    for message in messages:
        summary = per_session.setdefault(
            message.session_id,
            SessionSummary(session_id=message.session_id, messages=0, questions=0),
        )
        summary.messages += 1
        summary.questions += message.role == "user"

    return Metrics(
        sessions=len(per_session),
        messages=len(messages),
        questions=len(answers),
        answered=answered,
        avg_messages_per_session=len(messages) / len(per_session) if per_session else 0.0,
        no_answer_rate=(len(answers) - answered) / len(answers) if answers else 0.0,
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        avg_top_rerank_score=sum(top_scores) / len(top_scores) if top_scores else None,
        estimated_hours_saved=answered * manual_search_minutes / 60,
        manual_search_minutes=manual_search_minutes,
        top_urls=[
            UrlCount(url=url, count=count) for url, count in url_counts.most_common(TOP_URLS)
        ],
        messages_per_session=sorted(
            per_session.values(), key=lambda summary: summary.messages, reverse=True
        ),
    )
