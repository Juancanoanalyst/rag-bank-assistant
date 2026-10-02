from datetime import UTC, datetime

import pytest

from rag_assistant.analytics.metrics import compute_metrics, percentile
from rag_assistant.models import StoredMessage

NOW = datetime(2026, 4, 10, 15, 0, tzinfo=UTC)
CDT = "https://www.banco-ejemplo.com.co/cdt.html"
AHORROS = "https://www.banco-ejemplo.com.co/ahorros.html"


def exchange(session_id, latency_ms, answered, urls, scores) -> list[StoredMessage]:
    return [
        StoredMessage(session_id=session_id, role="user", content="pregunta", timestamp=NOW),
        StoredMessage(
            session_id=session_id,
            role="assistant",
            content="respuesta",
            timestamp=NOW,
            latency_ms=latency_ms,
            retrieved_urls=urls,
            rerank_scores=scores,
            answered=answered,
        ),
    ]


@pytest.fixture
def messages():
    return [
        *exchange("s1", 1000.0, True, [CDT, AHORROS], [0.8, 0.4]),
        *exchange("s1", 2000.0, True, [CDT], [0.6]),
        *exchange("s1", 3000.0, False, [AHORROS], [0.05]),
        *exchange("s2", 4000.0, True, [CDT], [0.7]),
    ]


def test_counts_sessions_messages_and_questions(messages):
    metrics = compute_metrics(messages, manual_search_minutes=5)

    assert (metrics.sessions, metrics.messages, metrics.questions, metrics.answered) == (2, 8, 4, 3)
    assert metrics.avg_messages_per_session == 4.0
    assert [(s.session_id, s.messages, s.questions) for s in metrics.messages_per_session] == [
        ("s1", 6, 3),
        ("s2", 2, 1),
    ]


def test_no_answer_rate_and_hours_saved(messages):
    metrics = compute_metrics(messages, manual_search_minutes=5)

    assert metrics.no_answer_rate == 0.25
    assert metrics.estimated_hours_saved == pytest.approx(3 * 5 / 60)
    assert metrics.manual_search_minutes == 5


def test_latency_percentiles(messages):
    metrics = compute_metrics(messages, manual_search_minutes=5)

    assert metrics.latency_p50_ms == 2500.0
    assert metrics.latency_p95_ms == pytest.approx(3850.0)


def test_top_urls_and_average_top_rerank_score(messages):
    metrics = compute_metrics(messages, manual_search_minutes=5)

    assert [(item.url, item.count) for item in metrics.top_urls] == [(CDT, 3), (AHORROS, 2)]
    assert metrics.avg_top_rerank_score == pytest.approx((0.8 + 0.6 + 0.05 + 0.7) / 4)


def test_empty_history_produces_zeroes_not_errors():
    metrics = compute_metrics([], manual_search_minutes=5)

    assert (metrics.sessions, metrics.messages, metrics.questions) == (0, 0, 0)
    assert (metrics.no_answer_rate, metrics.estimated_hours_saved) == (0.0, 0.0)
    assert metrics.latency_p50_ms is None
    assert metrics.avg_top_rerank_score is None
    assert metrics.top_urls == []


def test_answers_without_reranker_scores_are_left_out_of_the_average():
    metrics = compute_metrics(exchange("s1", 900.0, True, [CDT], []), manual_search_minutes=5)

    assert metrics.avg_top_rerank_score is None
    assert metrics.answered == 1


@pytest.mark.parametrize(
    "values, fraction, expected",
    [([7.0], 0.95, 7.0), ([1.0, 2.0], 0.5, 1.5), ([3.0, 1.0, 2.0], 0.5, 2.0), ([], 0.5, None)],
)
def test_percentile(values, fraction, expected):
    assert percentile(values, fraction) == expected
