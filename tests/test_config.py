import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_assistant.config import Settings, get_settings
from rag_assistant.exceptions import ConfigurationError

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Drop exported settings so tests behave the same in a shell, CI or the container."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


@pytest.mark.parametrize("field, value", [("api_port", 70000), ("chunk_size", 0), ("top_k", 0)])
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


@pytest.mark.parametrize("key", [None, "", "   "])
def test_groq_requires_api_key(key):
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        make_settings(llm_provider="groq", groq_api_key=key)


def test_groq_api_key_is_not_exposed_in_repr():
    settings = make_settings(llm_provider="groq", groq_api_key="test-key")

    assert "test-key" not in repr(settings)
    assert settings.groq_api_key.get_secret_value() == "test-key"


def test_unknown_llm_provider_is_rejected():
    with pytest.raises(ValidationError):
        make_settings(llm_provider="openai")


def test_get_settings_wraps_validation_errors(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("CHUNK_SIZE", "0")

    with pytest.raises(ConfigurationError):
        get_settings()


def test_get_settings_is_cached(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert get_settings() is get_settings()


def test_env_example_declares_every_setting():
    declared = {
        match.group(1).lower()
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"^([A-Z_]+)=", line))
    }

    assert declared == set(Settings.model_fields)


def test_env_example_values_are_valid():
    settings = Settings(_env_file=ENV_EXAMPLE)

    assert settings.qdrant_url == "http://qdrant:6333"
