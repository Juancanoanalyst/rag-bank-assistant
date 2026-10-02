import pytest
from fakes import FakeLLM

from rag_assistant import factory
from rag_assistant.exceptions import LLMError, RetrievalError
from rag_assistant.history import SQLiteHistoryRepository
from rag_assistant.models import Chunk, RetrievedChunk
from rag_assistant.service import prompts
from rag_assistant.service.rag_service import RAGService

SITE = "https://www.banco-ejemplo.com.co"


class FakeRetriever:
    """Returns preset chunks and records the queries it was asked."""

    def __init__(self, chunks: list[RetrievedChunk] | Exception) -> None:
        self._chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        if isinstance(self._chunks, Exception):
            raise self._chunks
        return self._chunks


def retrieved(page: str, text: str, rerank_score: float | None) -> RetrievedChunk:
    chunk = Chunk(
        id=f"{page}-{text}",
        url=f"{SITE}/{page}.html",
        title=page,
        section="personas",
        chunk_index=0,
        text=text,
    )
    return RetrievedChunk(chunk=chunk, vector_score=0.7, rerank_score=rerank_score)


CDT_CHUNKS = [
    retrieved("cdt", "El CDT es un depósito a término fijo.", 0.82),
    retrieved("cdt", "Puedes abrirlo desde un millón de pesos.", 0.55),
    retrieved("inversion", "Otras opciones de inversión.", 0.21),
]


@pytest.fixture
def history(tmp_path):
    return SQLiteHistoryRepository(tmp_path / "history.db")


def make_service(history, llm, chunks=CDT_CHUNKS, max_messages=6, min_score=0.15):
    retriever = FakeRetriever(chunks)
    service = RAGService(retriever, llm, history, max_messages, min_score)
    return service, retriever


# --- answering ---------------------------------------------------------------


def test_first_question_is_answered_from_the_retrieved_context(history):
    llm = FakeLLM("Un CDT es un depósito a término fijo.")
    service, retriever = make_service(history, llm)

    result = service.ask("s1", "¿Qué es un CDT?")

    system, user = llm.calls[0]
    assert result.answer == "Un CDT es un depósito a término fijo."
    assert result.answered
    assert result.sources == [f"{SITE}/cdt.html", f"{SITE}/inversion.html"]
    assert result.rerank_scores == [0.82, 0.55, 0.21]
    assert result.latency_ms > 0
    assert retriever.queries == ["¿Qué es un CDT?"]
    assert len(llm.calls) == 1  # nothing to condense without history
    assert system.role == "system" and prompts.NO_ANSWER_MARKER in system.content
    assert "El CDT es un depósito a término fijo." in user.content
    assert f"{SITE}/cdt.html" in user.content
    assert user.content.endswith("PREGUNTA: ¿Qué es un CDT?")


def test_exchange_is_persisted_with_metrics_on_the_assistant_message(history):
    service, _ = make_service(history, FakeLLM("Un CDT es un depósito."))

    result = service.ask("s1", "¿Qué es un CDT?")

    question, answer = history.session("s1")
    assert (question.role, question.content) == ("user", "¿Qué es un CDT?")
    assert (question.latency_ms, question.answered, question.retrieved_urls) == (None, None, [])
    assert (answer.role, answer.content) == ("assistant", "Un CDT es un depósito.")
    assert answer.latency_ms == result.latency_ms
    assert answer.retrieved_urls == [f"{SITE}/cdt.html", f"{SITE}/inversion.html"]
    assert answer.rerank_scores == [0.82, 0.55, 0.21]
    assert answer.answered is True
    assert question.timestamp <= answer.timestamp


# --- conversation memory -----------------------------------------------------


def test_follow_up_is_condensed_with_history_before_retrieval(history):
    llm = FakeLLM(
        "Un CDT es un depósito a término.",
        "¿Cuál es el monto mínimo para abrir un CDT?",
        "Desde un millón de pesos.",
    )
    service, retriever = make_service(history, llm)
    service.ask("s1", "¿Qué es un CDT?")

    result = service.ask("s1", "¿y cuál es el monto mínimo?")

    condense_call, answer_call = llm.calls[1], llm.calls[2]
    assert retriever.queries[-1] == "¿Cuál es el monto mínimo para abrir un CDT?"
    assert result.standalone_question == "¿Cuál es el monto mínimo para abrir un CDT?"
    assert result.answer == "Desde un millón de pesos."
    assert "Usuario: ¿Qué es un CDT?" in condense_call[-1].content
    assert "Asistente: Un CDT es un depósito a término." in condense_call[-1].content
    assert condense_call[-1].content.endswith("ÚLTIMA PREGUNTA: ¿y cuál es el monto mínimo?")
    # The answer prompt carries the previous turns and the question as the user wrote it.
    assert [message.role for message in answer_call] == ["system", "user", "assistant", "user"]
    assert answer_call[-1].content.endswith("PREGUNTA: ¿y cuál es el monto mínimo?")
    assert [item.content for item in history.session("s1")][2] == "¿y cuál es el monto mínimo?"


def test_only_the_last_n_messages_are_used(history):
    first, _ = make_service(history, FakeLLM("respuesta uno"), max_messages=2)
    first.ask("s1", "pregunta uno")
    second, _ = make_service(history, FakeLLM("condensada", "respuesta dos"), max_messages=2)
    second.ask("s1", "pregunta dos")
    llm = FakeLLM("condensada", "respuesta tres")
    service, _ = make_service(history, llm, max_messages=2)

    service.ask("s1", "pregunta tres")

    transcript = llm.calls[0][-1].content
    assert "pregunta dos" in transcript and "respuesta dos" in transcript
    assert "pregunta uno" not in transcript


def test_history_disabled_with_zero_messages(history):
    service, retriever = make_service(history, FakeLLM("r1"), max_messages=0)
    service.ask("s1", "pregunta uno")
    llm = FakeLLM("r2")
    service, retriever = make_service(history, llm, max_messages=0)

    service.ask("s1", "¿y eso?")

    assert retriever.queries == ["¿y eso?"]
    assert len(llm.calls) == 1
    assert len(history.session("s1")) == 4  # still persisted


def test_sessions_do_not_share_context(history):
    service, _ = make_service(history, FakeLLM("respuesta uno"))
    service.ask("s1", "¿Qué es un CDT?")
    llm = FakeLLM("respuesta dos")
    service, retriever = make_service(history, llm)

    service.ask("s2", "¿y el monto mínimo?")

    assert retriever.queries == ["¿y el monto mínimo?"]
    assert len(llm.calls) == 1


def test_condensing_failure_falls_back_to_the_original_question(history):
    service, _ = make_service(history, FakeLLM("respuesta uno"))
    service.ask("s1", "¿Qué es un CDT?")
    llm = FakeLLM(LLMError("timeout"), "Desde un millón de pesos.")
    service, retriever = make_service(history, llm)

    result = service.ask("s1", "¿y el monto mínimo?")

    assert retriever.queries == ["¿y el monto mínimo?"]
    assert result.answered
    assert result.answer == "Desde un millón de pesos."


# --- no answer found ---------------------------------------------------------


def test_low_rerank_score_skips_the_llm(history):
    llm = FakeLLM()
    chunks = [retrieved("historia", "Texto sin relación.", 0.06)]
    service, _ = make_service(history, llm, chunks=chunks)

    result = service.ask("s1", "¿Cuál es la capital de Francia?")

    assert not result.answered
    assert result.answer == prompts.NO_ANSWER_MESSAGE
    assert result.sources == []
    assert llm.calls == []
    assert history.session("s1")[1].answered is False
    assert history.session("s1")[1].retrieved_urls == [f"{SITE}/historia.html"]


def test_nothing_retrieved_is_no_answer(history):
    llm = FakeLLM()
    service, _ = make_service(history, llm, chunks=[])

    result = service.ask("s1", "¿algo?")

    assert (result.answered, result.answer) == (False, prompts.NO_ANSWER_MESSAGE)
    assert llm.calls == []


@pytest.mark.parametrize("reply", ["NO_ENCONTRADO", "no_encontrado.", "Lo siento. NO_ENCONTRADO"])
def test_llm_can_declare_that_the_context_has_no_answer(history, reply):
    service, _ = make_service(history, FakeLLM(reply))

    result = service.ask("s1", "¿Cuál es la tasa del CDT a 90 días?")

    assert not result.answered
    assert result.answer == prompts.NO_ANSWER_MESSAGE
    assert result.sources == []
    assert history.session("s1")[1].content == prompts.NO_ANSWER_MESSAGE


def test_without_a_reranker_the_llm_decides(history):
    llm = FakeLLM("Un CDT es un depósito.")
    chunks = [retrieved("cdt", "El CDT es un depósito a término fijo.", None)]
    service, _ = make_service(history, llm, chunks=chunks)

    result = service.ask("s1", "¿Qué es un CDT?")

    assert result.answered
    assert result.rerank_scores == []


# --- failures ----------------------------------------------------------------


def test_llm_failure_propagates_and_nothing_is_saved(history):
    service, _ = make_service(history, FakeLLM(LLMError("Ollama returned HTTP 500")))

    with pytest.raises(LLMError, match="HTTP 500"):
        service.ask("s1", "¿Qué es un CDT?")

    assert history.session("s1") == []


def test_retrieval_failure_propagates_and_nothing_is_saved(history):
    service, _ = make_service(history, FakeLLM(), chunks=RetrievalError("Vector search failed"))

    with pytest.raises(RetrievalError):
        service.ask("s1", "¿Qué es un CDT?")

    assert history.session("s1") == []


# --- factory -----------------------------------------------------------------


def test_factory_wires_the_service_from_settings(settings, tmp_path):
    custom = settings.model_copy(
        update={
            "history_db_path": tmp_path / "h" / "history.db",
            "history_max_messages": 4,
            "min_rerank_score": 0.3,
        }
    )

    service = factory.build_rag_service(custom)

    assert isinstance(service, RAGService)
    assert (service._history_max_messages, service._min_rerank_score) == (4, 0.3)
    assert (tmp_path / "h" / "history.db").exists()
