"""Streamlit UI: streamlit run src/rag_assistant/ui/app.py."""

import re
import uuid

import streamlit as st

from rag_assistant.config import get_settings
from rag_assistant.ui.api_client import ApiClient, ApiError

SESSION_ID_PATTERN = re.compile(r"^[\w.-]{1,64}$")


@st.cache_resource
def get_client() -> ApiClient:
    settings = get_settings()
    # The API waits for the LLM, so allow a little more than the LLM timeout.
    return ApiClient(settings.api_url, timeout=settings.llm_timeout_seconds + 30)


def new_session_id() -> str:
    return f"sesion-{uuid.uuid4().hex[:8]}"


def render_message(role: str, content: str, sources: list[str], caption: str | None) -> None:
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander(f"Fuentes ({len(sources)})"):
                for url in sources:
                    st.markdown(f"- [{url}]({url})")
        if caption:
            st.caption(caption)


def render_stored(message: dict) -> None:
    answered = message.get("answered")
    latency = message.get("latency_ms")
    render_message(
        message["role"],
        message["content"],
        sources=message.get("retrieved_urls", []) if answered else [],
        caption=f"{latency / 1000:.1f} s" if latency else None,
    )


def chat_tab(client: ApiClient, session_id: str) -> None:
    try:
        history = client.history(session_id)
    except ApiError as exc:
        st.error(str(exc))
        return

    if not history:
        st.info("Esta sesión no tiene mensajes. Escribe una pregunta para empezar.")
    for message in history:
        render_stored(message)

    question = st.chat_input("Pregunta sobre el contenido del sitio web del banco")
    if not question:
        return

    render_message("user", question, sources=[], caption=None)
    with st.spinner("Buscando en el contenido del sitio..."):
        try:
            result = client.chat(session_id, question)
        except ApiError as exc:
            st.error(str(exc))
            return
    render_message(
        "assistant",
        result["answer"],
        sources=result["sources"],
        caption=f"{result['latency_ms'] / 1000:.1f} s",
    )


def main() -> None:
    st.set_page_config(page_title="Asistente BBVA Colombia", page_icon="💬")
    st.title("Asistente del sitio web BBVA Colombia")

    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()

    with st.sidebar:
        st.header("Sesión")
        st.text_input(
            "ID de sesión",
            key="session_id",
            help="Usa el mismo ID para retomar una conversación anterior.",
        )
        # A callback, because a widget's state cannot be changed after it is drawn.
        st.button(
            "Nueva sesión",
            on_click=lambda: st.session_state.update(session_id=new_session_id()),
        )
        st.caption("El historial se guarda por ID de sesión.")

    session_id = st.session_state.session_id.strip()
    (chat,) = st.tabs(["Chat"])
    with chat:
        if SESSION_ID_PATTERN.match(session_id):
            chat_tab(get_client(), session_id)
        else:
            st.warning(
                "El ID de sesión solo admite letras, números, guiones, puntos y guion bajo "
                "(máximo 64 caracteres)."
            )


main()
