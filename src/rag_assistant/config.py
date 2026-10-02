"""Application settings, loaded from environment variables and an optional .env file.

This is the single place where tunable parameters live. Components never read
os.environ directly: they receive a Settings instance (see the factories).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_assistant.exceptions import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Scraping ---
    scraper_base_url: str = "https://www.bbva.com.co"
    scraper_sitemap_url: str = "https://www.bbva.com.co/sitemap.xml"
    scraper_max_pages: int = Field(default=150, gt=0)
    scraper_delay_seconds: float = Field(default=1.0, ge=0)
    scraper_timeout_seconds: float = Field(default=15.0, gt=0)
    scraper_user_agent: str = "rag-bank-assistant/0.1 (prueba tecnica; uso educativo)"
    raw_data_dir: Path = Path("data/raw")
    clean_data_dir: Path = Path("data/clean")

    # --- Chunking ---
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)

    # --- Embeddings / reranker (fastembed, ONNX) ---
    embedder_provider: Literal["fastembed"] = "fastembed"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_provider: Literal["fastembed", "none"] = "fastembed"
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"

    # --- Vector store ---
    # Service URLs default to localhost for runs outside Docker; .env.example
    # points them at the compose service names.
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "bank_site"

    # --- Retrieval ---
    top_k: int = Field(default=20, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)

    # --- LLM ---
    llm_provider: Literal["ollama", "groq"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)

    # --- Conversation history ---
    history_db_path: Path = Path("data/history/history.db")
    history_max_messages: int = Field(default=6, ge=0)

    # --- Analytics ---
    manual_search_minutes: float = Field(default=5.0, ge=0)

    # --- API / UI ---
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, gt=0, lt=65536)
    api_url: str = "http://localhost:8000"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.rerank_top_n > self.top_k:
            raise ValueError("RERANK_TOP_N cannot be greater than TOP_K")
        groq_key = self.groq_api_key.get_secret_value() if self.groq_api_key else ""
        if self.llm_provider == "groq" and not groq_key.strip():
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance.

    Relative paths (.env, data dirs) resolve against the working directory, so
    entrypoints are expected to run from the repository root (or /app in Docker).
    """
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc
