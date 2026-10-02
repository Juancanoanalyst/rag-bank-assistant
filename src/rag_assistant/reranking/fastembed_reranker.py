"""Cross-encoder reranking with fastembed (ONNX runtime, CPU, no PyTorch)."""

import logging
import math
import threading
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag_assistant.exceptions import ConfigurationError, RetrievalError
from rag_assistant.models import RetrievedChunk
from rag_assistant.reranking.base import Reranker

logger = logging.getLogger(__name__)


def _sigmoid(logit: float) -> float:
    # Clamped so extreme logits cannot overflow math.exp.
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))


class FastEmbedReranker(Reranker):
    """Scores each (question, chunk) pair jointly, which a vector search cannot do.

    Raw model outputs are logits; they are mapped to [0, 1] so the score can be
    compared against a threshold and averaged in the analytics.
    """

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        supported = {model["model"] for model in TextCrossEncoder.list_supported_models()}
        if model_name not in supported:
            raise ConfigurationError(
                f"RERANKER_MODEL={model_name!r} is not supported by fastembed. "
                "See TextCrossEncoder.list_supported_models() for valid names."
            )
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextCrossEncoder | None = None
        self._load_lock = threading.Lock()

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        try:
            logits = list(self._load().rerank(query, [item.chunk.text for item in candidates]))
        except Exception as exc:
            raise RetrievalError(f"Reranking {len(candidates)} chunks failed: {exc}") from exc

        scored = [
            item.model_copy(update={"rerank_score": _sigmoid(float(logit))})
            for item, logit in zip(candidates, logits, strict=True)
        ]
        scored.sort(key=lambda item: item.rerank_score, reverse=True)
        return scored[:top_n]

    def _load(self) -> TextCrossEncoder:
        with self._load_lock:
            if self._model is None:
                logger.info("Loading reranker model %s", self._model_name)
                cache_dir = str(self._cache_dir) if self._cache_dir else None
                self._model = TextCrossEncoder(self._model_name, cache_dir=cache_dir)
            return self._model
