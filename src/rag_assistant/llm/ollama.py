"""Local LLM served by Ollama."""

import requests

from rag_assistant.exceptions import LLMError
from rag_assistant.llm.base import LLMClient
from rag_assistant.llm.http import post_json, require_text
from rag_assistant.models import ChatMessage


class OllamaLLM(LLMClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        session: requests.Session | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model
        self._options = {"temperature": temperature, "num_predict": max_tokens}
        self._timeout = timeout
        self._session = session or requests.Session()

    def generate(self, messages: list[ChatMessage]) -> str:
        payload = {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            "options": self._options,
        }
        body = post_json(self._session, self._url, payload, self._timeout, provider="Ollama")
        try:
            content = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("Ollama returned an unexpected response shape") from exc
        return require_text(content, provider="Ollama")
