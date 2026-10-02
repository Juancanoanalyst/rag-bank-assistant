"""Embeddings with fastembed (ONNX runtime, CPU, no PyTorch)."""

import logging
import threading
from pathlib import Path

from fastembed import TextEmbedding

from rag_assistant.embeddings.base import Embedder
from rag_assistant.exceptions import ConfigurationError, EmbeddingError

logger = logging.getLogger(__name__)


class FastEmbedEmbedder(Embedder):
    def __init__(
        self, model_name: str, cache_dir: Path | None = None, batch_size: int = 32
    ) -> None:
        supported = {model["model"] for model in TextEmbedding.list_supported_models()}
        if model_name not in supported:
            raise ConfigurationError(
                f"EMBEDDING_MODEL={model_name!r} is not supported by fastembed. "
                "See TextEmbedding.list_supported_models() for valid names."
            )
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model: TextEmbedding | None = None
        self._load_lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return TextEmbedding.get_embedding_size(self._model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._load().passage_embed(texts, batch_size=self._batch_size)
            return [vector.tolist() for vector in vectors]
        except Exception as exc:
            raise EmbeddingError(f"Embedding {len(texts)} passages failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return next(iter(self._load().query_embed([text]))).tolist()
        except Exception as exc:
            raise EmbeddingError(f"Embedding the query failed: {exc}") from exc

    def _load(self) -> TextEmbedding:
        # Loaded on first use: the model download (hundreds of MB) should not
        # happen just because the object was constructed. The lock stops
        # concurrent first requests from each loading their own copy.
        with self._load_lock:
            if self._model is None:
                logger.info("Loading embedding model %s", self._model_name)
                cache_dir = str(self._cache_dir) if self._cache_dir else None
                self._model = TextEmbedding(self._model_name, cache_dir=cache_dir)
            return self._model
