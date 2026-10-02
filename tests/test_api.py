import threading
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api import __main__ as api_main
from rag_assistant.api import app as api_app
from rag_assistant.api.app import create_app
from rag_assistant.exceptions import (
    ConfigurationError,
    EmbeddingError,
    HistoryError,
    LLMError,
    RetrievalError,
)
from rag_assistant.history import SQLiteHistoryRepository
from rag_assistant.models import ChatResult, StoredMessage

SOURCE = "https://www.banco-ejemplo.com.co/cdt.html"


class StubService:
    """Stands in for RAGService: returns a fixed result or raises a preset error."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.asked: list[tuple[str, str]] = []

    def ask(self, session_id: str, question: str) -> ChatResult:
        self.asked.append((session_id, question))
        if self.error:
            raise self.error
        return ChatResult(
            session_id=session_id,
            answer="Un CDT es un depósito a término.",
            answered=True,
            sources=[SOURCE],
            rerank_scores=[0.82],
            standalone_question=question,
            latency_ms=1234.5,
        )


@pytest.fixture
def service():
    return StubService()


@pytest.fixture
def history(tmp_path):
    return SQLiteHistoryRepository(tmp_path / "history.db")


@pytest.fixture
def client(service, history):
    with TestClient(create_app(service=service, history=history)) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_the_answer_with_sources_and_metrics(client, service):
    response = client.post("/chat", json={"session_id": "s1", "question": "¿Qué es un CDT?"})

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "s1",
        "answer": "Un CDT es un depósito a término.",
        "answered": True,
        "sources": [SOURCE],
        "rerank_scores": [0.82],
        "standalone_question": "¿Qué es un CDT?",
        "latency_ms": 1234.5,
    }
    assert service.asked == [("s1", "¿Qué es un CDT?")]


def test_chat_strips_surrounding_whitespace(client, service):
    client.post("/chat", json={"session_id": "  s1 ", "question": "  ¿Qué es un CDT?\n"})

    assert service.asked == [("s1", "¿Qué es un CDT?")]


@pytest.mark.parametrize(
    "body",
    [
        {"session_id": "s1"},
        {"question": "hola"},
        {"session_id": "s1", "question": "   "},
        {"session_id": "", "question": "hola"},
        {"session_id": "con espacios", "question": "hola"},
        {"session_id": "a/b", "question": "hola"},
        {"session_id": "x" * 65, "question": "hola"},
        {"session_id": "s1", "question": "x" * 2001},
    ],
)
def test_chat_rejects_invalid_input_without_calling_the_service(client, service, body):
    response = client.post("/chat", json=body)

    assert response.status_code == 422
    assert service.asked == []


@pytest.mark.parametrize(
    "error, status",
    [
        (LLMError("Ollama returned HTTP 500: secret-internal-detail"), 502),
        (RetrievalError("Vector search failed: secret-internal-detail"), 503),
        (EmbeddingError("onnx: secret-internal-detail"), 503),
        (HistoryError("database is locked: secret-internal-detail"), 500),
        (ConfigurationError("secret-internal-detail"), 500),
    ],
)
def test_domain_errors_map_to_status_codes_without_leaking_details(client, service, error, status):
    service.error = error

    response = client.post("/chat", json={"session_id": "s1", "question": "hola"})

    assert response.status_code == status
    assert "secret-internal-detail" not in response.text
    assert response.json()["detail"]


def test_history_returns_the_messages_of_a_session(client, history):
    asked_at = datetime(2026, 4, 10, 15, 0, tzinfo=UTC)
    history.add(
        [
            StoredMessage(
                session_id="s1", role="user", content="¿Qué es un CDT?", timestamp=asked_at
            ),
            StoredMessage(
                session_id="s1",
                role="assistant",
                content="Un CDT es un depósito.",
                timestamp=asked_at,
                latency_ms=1500.0,
                retrieved_urls=[SOURCE],
                rerank_scores=[0.82],
                answered=True,
            ),
            StoredMessage(session_id="s2", role="user", content="otra", timestamp=asked_at),
        ]
    )

    response = client.get("/sessions/s1/history")

    body = response.json()
    assert response.status_code == 200
    assert body["session_id"] == "s1"
    assert [(m["role"], m["content"]) for m in body["messages"]] == [
        ("user", "¿Qué es un CDT?"),
        ("assistant", "Un CDT es un depósito."),
    ]
    assert body["messages"][1]["answered"] is True
    assert body["messages"][1]["latency_ms"] == 1500.0
    assert body["messages"][1]["retrieved_urls"] == [SOURCE]
    assert body["messages"][1]["timestamp"].startswith("2026-04-10T15:00:00")


def test_history_of_unknown_session_is_empty(client):
    response = client.get("/sessions/desconocida/history")

    assert response.status_code == 200
    assert response.json() == {"session_id": "desconocida", "messages": []}


def test_history_rejects_invalid_session_id(client):
    assert client.get("/sessions/con espacios/history").status_code == 422


def test_history_failure_is_reported_as_500(service, tmp_path):
    class BrokenHistory(SQLiteHistoryRepository):
        def session(self, session_id):
            raise HistoryError("database is locked")

    app = create_app(service=service, history=BrokenHistory(tmp_path / "h.db"))
    with TestClient(app) as client:
        response = client.get("/sessions/s1/history")

    assert response.status_code == 500
    assert "locked" not in response.text


def test_metrics_summarise_the_stored_history(service, history):
    asked_at = datetime(2026, 4, 10, 15, 0, tzinfo=UTC)
    history.add(
        [
            StoredMessage(session_id="s1", role="user", content="p", timestamp=asked_at),
            StoredMessage(
                session_id="s1",
                role="assistant",
                content="r",
                timestamp=asked_at,
                latency_ms=2000.0,
                retrieved_urls=[SOURCE],
                rerank_scores=[0.8],
                answered=True,
            ),
        ]
    )
    app = create_app(service=service, history=history, manual_search_minutes=12)

    with TestClient(app) as client:
        response = client.get("/metrics")

    body = response.json()
    assert response.status_code == 200
    assert (body["sessions"], body["questions"], body["answered"]) == (1, 1, 1)
    assert body["latency_p50_ms"] == 2000.0
    assert body["estimated_hours_saved"] == 0.2
    assert body["top_urls"] == [{"url": SOURCE, "count": 1}]


def test_metrics_with_no_history(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["questions"] == 0


# --- production wiring (no stubs passed to create_app) -----------------------


class WarmableService(StubService):
    def __init__(self, warm_up_error: Exception | None = None) -> None:
        super().__init__()
        self.warm_up_error = warm_up_error
        self.warmed = threading.Event()

    def warm_up(self) -> None:
        self.warmed.set()
        if self.warm_up_error:
            raise self.warm_up_error


@pytest.mark.parametrize("warm_up_error", [None, RetrievalError("qdrant is still starting")])
def test_app_builds_itself_from_settings_and_warms_up(
    settings, history, monkeypatch, warm_up_error
):
    built = WarmableService(warm_up_error)
    custom = settings.model_copy(update={"manual_search_minutes": 9})
    monkeypatch.setattr(api_app, "get_settings", lambda: custom)
    monkeypatch.setattr(api_app, "build_rag_service", lambda _: built)
    monkeypatch.setattr(api_app, "build_history", lambda _: history)

    with TestClient(create_app()) as client:
        chat = client.post("/chat", json={"session_id": "s1", "question": "hola"})
        metrics = client.get("/metrics")

    assert built.warmed.wait(timeout=5)
    assert chat.status_code == 200  # a failed warm-up does not take the API down
    assert metrics.json()["manual_search_minutes"] == 9


def test_api_entrypoint_runs_uvicorn_with_configured_host_and_port(settings, monkeypatch):
    calls = []
    custom = settings.model_copy(update={"api_host": "127.0.0.1", "api_port": 9000})
    monkeypatch.setattr(api_main, "get_settings", lambda: custom)
    monkeypatch.setattr(
        api_main.uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    api_main.main()

    assert calls == [
        (
            ("rag_assistant.api.app:create_app",),
            {"factory": True, "host": "127.0.0.1", "port": 9000, "log_level": "info"},
        )
    ]
