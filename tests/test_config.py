import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_assistant.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def make_settings(**overrides) -> Settings:
    # _env_file=None keeps tests independent from a developer's local .env.
    return Settings(_env_file=None, **overrides)


def test_defaults_are_valid():
    settings = make_settings()

    assert settings.scraper_max_pages == 150
    assert settings.llm_provider == "ollama"
    assert settings.ollama_model == "qwen2.5:3b"
    assert settings.rerank_top_n <= settings.top_k


def test_values_come_from_environment(monkeypatch):
    monkeypatch.setenv("HISTORY_MAX_MESSAGES", "10")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")

    settings = make_settings()

    assert settings.history_max_messages == 10
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        make_settings(chunk_size=200, chunk_overlap=200)


def test_rerank_top_n_cannot_exceed_top_k():
    with pytest.raises(ValidationError, match="RERANK_TOP_N"):
        make_settings(top_k=3, rerank_top_n=5)


def test_groq_requires_api_key():
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        make_settings(llm_provider="groq", groq_api_key=None)

    assert make_settings(llm_provider="groq", groq_api_key="test-key").llm_provider == "groq"


def test_unknown_llm_provider_is_rejected():
    with pytest.raises(ValidationError):
        make_settings(llm_provider="openai")


def test_env_example_only_declares_known_settings():
    declared = {
        match.group(1).lower()
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"^([A-Z_]+)=", line))
    }

    unknown = declared - set(Settings.model_fields)
    assert not unknown, f".env.example declares settings that do not exist: {unknown}"
