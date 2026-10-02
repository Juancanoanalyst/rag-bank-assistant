import re
from pathlib import Path

import pytest
import requests
import responses
from streamlit.testing.v1 import AppTest

from rag_assistant.config import get_settings
from rag_assistant.ui.api_client import ApiClient, ApiError

API = "http://api.test"
APP_FILE = str(Path(__file__).resolve().parents[1] / "src" / "rag_assistant" / "ui" / "app.py")
CHAT_RESULT = {
    "session_id": "s1",
    "answer": "Un CDT es un depósito a término.",
    "answered": True,
    "sources": ["https://www.banco-ejemplo.com.co/cdt.html"],
    "rerank_scores": [0.82],
    "standalone_question": "¿Qué es un CDT?",
    "latency_ms": 2500.0,
}


@pytest.fixture
def client():
    return ApiClient(API, timeout=5)


# --- api client --------------------------------------------------------------


@responses.activate
def test_client_posts_the_question_and_returns_the_result(client):
    responses.post(f"{API}/chat", json=CHAT_RESULT)

    assert client.chat("s1", "¿Qué es un CDT?") == CHAT_RESULT
    assert responses.calls[0].request.body == (
        b'{"session_id": "s1", "question": "\\u00bfQu\\u00e9 es un CDT?"}'
    )


@responses.activate
def test_client_returns_the_messages_of_a_session(client):
    messages = [{"role": "user", "content": "hola"}]
    responses.get(f"{API}/sessions/s1/history", json={"session_id": "s1", "messages": messages})

    assert client.history("s1") == messages


@responses.activate
def test_client_shows_the_api_error_message(client):
    responses.post(f"{API}/chat", status=502, json={"detail": "El modelo no está disponible."})

    with pytest.raises(ApiError, match="El modelo no está disponible"):
        client.chat("s1", "hola")


@responses.activate
@pytest.mark.parametrize(
    "setup, message",
    [
        ({"body": requests.ConnectionError("refused")}, "No se pudo conectar"),
        ({"body": requests.Timeout("slow")}, "tardó demasiado"),
        ({"status": 422, "json": {"detail": [{"msg": "too long"}]}}, "no son válidos"),
        ({"status": 500, "body": "<html>error</html>"}, "HTTP 500"),
    ],
)
def test_client_turns_failures_into_readable_errors(client, setup, message):
    responses.post(f"{API}/chat", **setup)

    with pytest.raises(ApiError, match=message):
        client.chat("s1", "hola")


# --- streamlit app (runs headless, API mocked) -------------------------------


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("API_URL", API)
    get_settings.cache_clear()
    yield AppTest.from_file(APP_FILE, default_timeout=30)
    get_settings.cache_clear()


def history_url(app_test: AppTest) -> str:
    return f"{API}/sessions/{app_test.session_state.session_id}/history"


@responses.activate
def test_app_starts_with_a_generated_session_and_an_empty_chat(app):
    responses.get(
        re.compile(rf"{API}/sessions/sesion-[0-9a-f]{{8}}/history"),
        json={"session_id": "x", "messages": []},
    )

    app.run()

    assert not app.exception
    assert app.session_state.session_id.startswith("sesion-")
    assert "no tiene mensajes" in app.info[0].value


def test_app_shows_stored_history_and_sends_a_new_question(app):
    stored = [
        {"role": "user", "content": "¿Qué es un CDT?"},
        {
            "role": "assistant",
            "content": "Un CDT es un depósito a término.",
            "answered": True,
            "latency_ms": 2500.0,
            "retrieved_urls": ["https://www.banco-ejemplo.com.co/cdt.html"],
        },
    ]
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        mock.get(f"{API}/sessions/s1/history", json={"session_id": "s1", "messages": stored})
        mock.post(f"{API}/chat", json={**CHAT_RESULT, "answer": "Desde un millón de pesos."})
        app.session_state.session_id = "s1"
        app.run()
        shown_before = [message.markdown[0].value for message in app.chat_message]

        app.chat_input[0].set_value("¿y el monto mínimo?").run()

        sent = [call.request for call in mock.calls if call.request.method == "POST"]

    shown_after = [message.markdown[0].value for message in app.chat_message]
    assert not app.exception
    assert shown_before == ["¿Qué es un CDT?", "Un CDT es un depósito a término."]
    assert shown_after[-2:] == ["¿y el monto mínimo?", "Desde un millón de pesos."]
    assert len(sent) == 1 and b'"session_id": "s1"' in sent[0].body


def test_app_reports_an_unreachable_api_instead_of_crashing(app):
    with responses.RequestsMock() as mock:
        mock.get(f"{API}/sessions/s1/history", body=requests.ConnectionError("refused"))
        app.session_state.session_id = "s1"
        app.run()

    assert not app.exception
    assert "No se pudo conectar" in app.error[0].value


def test_app_rejects_an_invalid_session_id_without_calling_the_api(app):
    with responses.RequestsMock():
        app.session_state.session_id = "id con espacios"
        app.run()

    assert not app.exception
    assert "solo admite" in app.warning[0].value


# --- analytics tab -----------------------------------------------------------

METRICS = {
    "sessions": 2,
    "messages": 8,
    "questions": 4,
    "answered": 3,
    "avg_messages_per_session": 4.0,
    "no_answer_rate": 0.25,
    "latency_p50_ms": 2500.0,
    "latency_p95_ms": 3850.0,
    "avg_top_rerank_score": 0.54,
    "estimated_hours_saved": 0.25,
    "manual_search_minutes": 5.0,
    "top_urls": [{"url": "https://www.banco-ejemplo.com.co/cdt.html", "count": 3}],
    "messages_per_session": [{"session_id": "s1", "messages": 6, "questions": 3}],
}


def run_with_metrics(app, metrics_response: dict) -> AppTest:
    with responses.RequestsMock() as mock:
        mock.get(f"{API}/sessions/s1/history", json={"session_id": "s1", "messages": []})
        mock.get(f"{API}/metrics", **metrics_response)
        app.session_state.session_id = "s1"
        app.run()
    return app


def test_analytics_tab_shows_usage_quality_and_impact(app):
    run_with_metrics(app, {"json": METRICS})

    shown = {metric.label: metric.value for metric in app.metric}
    assert not app.exception
    assert shown == {
        "Sesiones": "2",
        "Preguntas": "4",
        "Mensajes por sesión": "4.0",
        "Latencia p50": "2.5 s",
        "Latencia p95": "3.9 s",
        "Sin respuesta": "25%",
        "Puntaje reranker": "0.54",
        "Horas ahorradas (estimado)": "0.2 h",
    }
    assert len(app.dataframe) == 2


def test_analytics_tab_with_no_conversations_yet(app):
    empty = {**METRICS, "questions": 0, "latency_p50_ms": None, "avg_top_rerank_score": None}

    run_with_metrics(app, {"json": empty})

    assert not app.exception
    assert any("Aún no hay conversaciones" in info.value for info in app.info)
    assert len(app.metric) == 0


def test_analytics_tab_reports_api_errors(app):
    run_with_metrics(app, {"status": 500, "json": {"detail": "No se pudo acceder al historial."}})

    assert not app.exception
    assert any("No se pudo acceder al historial" in error.value for error in app.error)
