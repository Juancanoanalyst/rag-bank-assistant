"""LLM interface (Strategy pattern).

RAGService only knows this interface, so the local model (Ollama) and the
hosted one (Groq) are interchangeable through LLM_PROVIDER.
"""

from abc import ABC, abstractmethod

from rag_assistant.models import ChatMessage


class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages: list[ChatMessage]) -> str:
        """Return the assistant's reply to a chat conversation."""
