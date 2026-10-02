"""Hosted LLM on Groq (OpenAI-compatible API). Optional: needs GROQ_API_KEY."""

import requests

from rag_assistant.exceptions import LLMError
from rag_assistant.llm.base import LLMClient
from rag_assistant.llm.http import post_json, require_text
from rag_assistant.models import ChatMessage

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        session: requests.Session | None = None,
    ) -> None:
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._session = session or requests.Session()

    def generate(self, messages: list[ChatMessage], max_tokens: int | None = None) -> str:
        payload = {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            "temperature": self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }
        body = post_json(
            self._session,
            GROQ_CHAT_URL,
            payload,
            self._timeout,
            provider="Groq",
            headers=self._headers,
        )
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Groq returned an unexpected response shape") from exc
        return require_text(content, provider="Groq")
