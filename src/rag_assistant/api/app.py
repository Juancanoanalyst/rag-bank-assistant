"""HTTP API: a thin layer over RAGService and the history repository."""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_assistant import __version__
from rag_assistant.analytics import Metrics, compute_metrics
from rag_assistant.api.schemas import ChatRequest, Health, SessionHistory, SessionId
from rag_assistant.config import get_settings
from rag_assistant.exceptions import (
    EmbeddingError,
    HistoryError,
    LLMError,
    RAGError,
    RetrievalError,
)
from rag_assistant.factory import build_history, build_rag_service
from rag_assistant.history import HistoryRepository
from rag_assistant.logging_config import configure_logging
from rag_assistant.models import ChatResult
from rag_assistant.service.rag_service import RAGService

logger = logging.getLogger(__name__)

# Domain errors -> HTTP status and a message that is safe to show to the user.
# The technical detail goes to the log, not to the response.
_ERROR_RESPONSES: list[tuple[type[RAGError], int, str]] = [
    (LLMError, 502, "El modelo de lenguaje no está disponible en este momento."),
    (RetrievalError, 503, "El buscador de contenido no está disponible o aún no se ha indexado."),
    (EmbeddingError, 503, "El modelo de embeddings no está disponible."),
    (HistoryError, 500, "No se pudo acceder al historial de conversaciones."),
]
_DEFAULT_ERROR = (500, "Ocurrió un error interno al procesar la solicitud.")


def _warm_up(service: RAGService) -> None:
    try:
        service.warm_up()
        logger.info("Models loaded")
    except RAGError as exc:
        logger.warning("Warm-up failed; models will load on the first question: %s", exc)


def create_app(
    service: RAGService | None = None,
    history: HistoryRepository | None = None,
    manual_search_minutes: float = 5.0,
) -> FastAPI:
    """Build the application.

    Tests pass `service` and `history` doubles. In production both are built
    from Settings when the server starts, so a configuration error stops the
    process at start-up instead of failing on the first request.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            settings = get_settings()
            configure_logging(settings.log_level)
            app.state.service = build_rag_service(settings)
            app.state.history = build_history(settings)
            app.state.manual_search_minutes = settings.manual_search_minutes
            # Loading the models takes tens of seconds; do it now, off the event
            # loop, rather than during the first user's question.
            threading.Thread(target=_warm_up, args=(app.state.service,), daemon=True).start()
        else:
            app.state.service = service
            app.state.history = history
            app.state.manual_search_minutes = manual_search_minutes
        yield

    app = FastAPI(title="RAG Bank Assistant", version=__version__, lifespan=lifespan)

    @app.exception_handler(RAGError)
    def handle_domain_error(request: Request, exc: RAGError) -> JSONResponse:
        status, message = next(
            ((code, text) for kind, code, text in _ERROR_RESPONSES if isinstance(exc, kind)),
            _DEFAULT_ERROR,
        )
        logger.error("%s %s failed: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=status, content={"detail": message})

    @app.get("/health")
    def health() -> Health:
        return Health(status="ok")

    # Plain `def` endpoints run in FastAPI's thread pool: the pipeline is
    # blocking (ONNX inference, HTTP to the LLM) and must not stall the event loop.
    @app.post("/chat")
    def chat(body: ChatRequest, request: Request) -> ChatResult:
        return request.app.state.service.ask(body.session_id, body.question)

    @app.get("/sessions/{session_id}/history")
    def session_history(session_id: SessionId, request: Request) -> SessionHistory:
        messages = request.app.state.history.session(session_id)
        return SessionHistory(session_id=session_id, messages=messages)

    @app.get("/metrics")
    def metrics(request: Request) -> Metrics:
        state = request.app.state
        return compute_metrics(state.history.all(), state.manual_search_minutes)

    return app
