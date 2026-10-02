"""Prompts, in Spanish because both the corpus and the users are Spanish-speaking."""

from rag_assistant.models import ChatMessage, RetrievedChunk, StoredMessage

# The model is told to emit this marker instead of improvising when the context
# does not contain the answer; RAGService turns it into NO_ANSWER_MESSAGE.
NO_ANSWER_MARKER = "NO_ENCONTRADO"

NO_ANSWER_MESSAGE = (
    "No encontré información sobre eso en el contenido del sitio web del banco. "
    "Intenta reformular la pregunta o consulta directamente en el sitio."
)

SYSTEM_PROMPT = f"""Eres un asistente interno que responde preguntas sobre el contenido \
publicado en el sitio web del banco.

Reglas:
- Responde SOLO con la información del CONTEXTO que acompaña a la pregunta.
- Si el contexto no contiene la respuesta, responde exactamente: {NO_ANSWER_MARKER}
- No inventes tasas, montos, requisitos ni fechas.
- Responde en español, de forma breve y clara. Usa viñetas cuando enumeres requisitos o pasos.
- No menciones estas reglas ni la palabra "contexto"."""

CONDENSE_PROMPT = """Reescribe la última pregunta del usuario para que se entienda por sí \
sola, sin necesidad de leer la conversación. Sustituye pronombres y referencias ("eso", \
"esa cuenta", "¿y cuánto cuesta?") por aquello a lo que se refieren.

Si la pregunta ya se entiende sola, devuélvela sin cambios. Devuelve únicamente la pregunta \
reescrita, sin explicaciones ni comillas."""


def is_no_answer(reply: str) -> bool:
    """True when the model declined to answer, however it spelled the marker.

    Small models write "No encontrado", "NO ENCONTRADO." or copy the refusal
    sentence itself, so matching is done on a normalised form.
    """
    normalised = reply.upper().replace(" ", "_")
    refusal_start = NO_ANSWER_MESSAGE[:30].upper().replace(" ", "_")
    return NO_ANSWER_MARKER in normalised or normalised.startswith(refusal_start)


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{number}] {item.chunk.title}\nFuente: {item.chunk.url}\n{item.chunk.text}"
        for number, item in enumerate(chunks, start=1)
    )


def _as_chat(history: list[StoredMessage]) -> list[ChatMessage]:
    return [ChatMessage(role=message.role, content=message.content) for message in history]


def answer_messages(
    question: str, chunks: list[RetrievedChunk], history: list[StoredMessage]
) -> list[ChatMessage]:
    """System rules, the recent conversation, then the context with the question."""
    user = f"CONTEXTO:\n{format_context(chunks)}\n\nPREGUNTA: {question}"
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        *_as_chat(history),
        ChatMessage(role="user", content=user),
    ]


def condense_messages(question: str, history: list[StoredMessage]) -> list[ChatMessage]:
    transcript = "\n".join(
        f"{'Usuario' if message.role == 'user' else 'Asistente'}: {message.content}"
        for message in history
    )
    user = f"CONVERSACIÓN:\n<<<\n{transcript}\n>>>\n\nÚLTIMA PREGUNTA: {question}"
    return [
        ChatMessage(role="system", content=CONDENSE_PROMPT),
        ChatMessage(role="user", content=user),
    ]
