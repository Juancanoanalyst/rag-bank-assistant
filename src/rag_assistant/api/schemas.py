"""Request and response bodies of the HTTP API."""

from typing import Annotated

from pydantic import BaseModel, StringConstraints

from rag_assistant.models import StoredMessage

# Session ids are chosen by the user in the UI; keep them short and free of
# characters that would be awkward in a URL path. The first character must be
# alphanumeric so that "." and ".." cannot be used as a path segment.
SessionId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][\w.-]*$"
    ),
]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ChatRequest(BaseModel):
    session_id: SessionId
    question: Question


class SessionHistory(BaseModel):
    session_id: str
    messages: list[StoredMessage]


class Health(BaseModel):
    status: str
