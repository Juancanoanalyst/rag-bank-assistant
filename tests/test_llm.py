import json

import pytest
import requests
import responses
from pydantic import SecretStr

from rag_assistant import factory
from rag_assistant.exceptions import ConfigurationError, LLMError
from rag_assistant.llm.groq import GROQ_CHAT_URL, GroqLLM
from rag_assistant.llm.ollama import OllamaLLM
from rag_assistant.models import ChatMessage

OLLAMA_URL = "http://ollama:11434/api/chat"
MESSAGES = [
    ChatMessage(role="system", content="Eres un asistente."),
    ChatMessage(role="user", content="¿Qué es un CDT?"),
]


def make_ollama(**overrides) -> OllamaLLM:
    options = {
        "base_url": "http://ollama:11434/",
        "model": "qwen2.5:3b",
        "temperature": 0.1,
        "max_tokens": 256,
        "timeout": 5,
        "num_ctx": 4096,
    }
    return OllamaLLM(**{**options, **overrides})


def make_groq() -> GroqLLM:
    return GroqLLM(
        api_key="test-key", model="llama-3.1-8b-instant", temperature=0.1, max_tokens=256, timeout=5
    )


def sent_payload() -> dict:
    return json.loads(responses.calls[0].request.body)


# --- ollama ------------------------------------------------------------------


@responses.activate
def test_ollama_sends_chat_request_and_returns_the_reply():
    responses.post(OLLAMA_URL, json={"message": {"role": "assistant", "content": " Un CDT es... "}})

    reply = make_ollama().generate(MESSAGES)

    assert reply == "Un CDT es..."
    assert sent_payload() == {
        "model": "qwen2.5:3b",
        "messages": [
            {"role": "system", "content": "Eres un asistente."},
            {"role": "user", "content": "¿Qué es un CDT?"},
        ],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.1, "num_predict": 256, "num_ctx": 4096},
    }


@responses.activate
def test_ollama_reports_a_missing_model_with_the_provider_message():
    responses.post(OLLAMA_URL, status=404, json={"error": "model 'qwen2.5:3b' not found"})

    with pytest.raises(LLMError, match="HTTP 404.*not found"):
        make_ollama().generate(MESSAGES)


@responses.activate
@pytest.mark.parametrize(
    "failure, message",
    [
        (requests.ConnectionError("connection refused"), "Could not reach Ollama"),
        (requests.Timeout("read timed out"), "did not answer within 5s"),
    ],
)
def test_ollama_network_failures_become_llm_errors(failure, message):
    responses.post(OLLAMA_URL, body=failure)

    with pytest.raises(LLMError, match=message):
        make_ollama().generate(MESSAGES)


@responses.activate
@pytest.mark.parametrize(
    "body",
    [{"unexpected": True}, {"message": {"content": "   "}}, {"message": {"content": None}}],
)
def test_ollama_rejects_unusable_responses(body):
    responses.post(OLLAMA_URL, json=body)

    with pytest.raises(LLMError):
        make_ollama().generate(MESSAGES)


@responses.activate
def test_ollama_rejects_non_json_responses():
    responses.post(OLLAMA_URL, body="<html>502 Bad Gateway</html>")

    with pytest.raises(LLMError, match="not JSON"):
        make_ollama().generate(MESSAGES)


# --- groq --------------------------------------------------------------------


@responses.activate
def test_groq_sends_openai_style_request_with_bearer_token():
    responses.post(GROQ_CHAT_URL, json={"choices": [{"message": {"content": "Un CDT es..."}}]})

    reply = make_groq().generate(MESSAGES)

    request = responses.calls[0].request
    assert reply == "Un CDT es..."
    assert request.headers["Authorization"] == "Bearer test-key"
    assert sent_payload()["model"] == "llama-3.1-8b-instant"
    assert sent_payload()["max_tokens"] == 256
    assert len(sent_payload()["messages"]) == 2


@responses.activate
def test_groq_errors_do_not_leak_the_api_key():
    responses.post(GROQ_CHAT_URL, status=401, json={"error": {"message": "Invalid API Key"}})

    with pytest.raises(LLMError, match="HTTP 401") as error:
        make_groq().generate(MESSAGES)

    assert "test-key" not in str(error.value)


@responses.activate
def test_groq_rejects_responses_without_choices():
    responses.post(GROQ_CHAT_URL, json={"choices": []})

    with pytest.raises(LLMError, match="unexpected response shape"):
        make_groq().generate(MESSAGES)


# --- factory -----------------------------------------------------------------


def test_factory_builds_ollama_by_default(settings):
    assert isinstance(factory.build_llm(settings), OllamaLLM)


def test_factory_builds_groq_when_configured(settings):
    key = SecretStr("test-key")
    custom = settings.model_copy(update={"llm_provider": "groq", "groq_api_key": key})

    assert isinstance(factory.build_llm(custom), GroqLLM)


def test_factory_refuses_groq_without_a_key(settings):
    custom = settings.model_copy(update={"llm_provider": "groq", "groq_api_key": None})

    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        factory.build_llm(custom)
