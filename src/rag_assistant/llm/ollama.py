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
        num_ctx: int,
        session: requests.Session | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_ctx = num_ctx
        self._timeout = timeout
        self._session = session or requests.Session()

    def generate(self, messages: list[ChatMessage], max_tokens: int | None = None) -> str:
        payload = {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            # Keep the model in memory between questions instead of reloading it.
            "keep_alive": "30m",
            "options": {
                "temperature": self._temperature,
                "num_predict": max_tokens or self._max_tokens,
                # Ollama's default window is small and truncates silently, which
                # would drop the system rules or the oldest context chunks.
                "num_ctx": self._num_ctx,
            },
        }
        body = post_json(self._session, self._url, payload, self._timeout, provider="Ollama")
        try:
            content = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("Ollama returned an unexpected response shape") from exc
        return require_text(content, provider="Ollama")
