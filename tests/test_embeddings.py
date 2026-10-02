import numpy as np
import pytest

from rag_assistant.embeddings import fastembed_embedder
from rag_assistant.embeddings.fastembed_embedder import FastEmbedEmbedder
from rag_assistant.exceptions import EmbeddingError

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class StubModel:
    """Stands in for fastembed.TextEmbedding so no model is downloaded."""

    def __init__(self) -> None:
        self.passage_calls: list[tuple[list[str], int]] = []

    def passage_embed(self, texts, batch_size):
        self.passage_calls.append((list(texts), batch_size))
        return iter(np.full(3, float(len(text))) for text in texts)

    def query_embed(self, texts):
        return iter([np.array([1.0, 2.0, 3.0])])


def embedder_with(model) -> FastEmbedEmbedder:
    embedder = FastEmbedEmbedder(MODEL, batch_size=8)
    embedder._model = model
    return embedder


def test_documents_are_embedded_as_passages_in_plain_lists():
    model = StubModel()

    vectors = embedder_with(model).embed_documents(["uno", "cuatro"])

    assert vectors == [[3.0, 3.0, 3.0], [6.0, 6.0, 6.0]]
    assert model.passage_calls == [(["uno", "cuatro"], 8)]


def test_query_is_embedded_as_a_plain_list():
    assert embedder_with(StubModel()).embed_query("¿qué es un CDT?") == [1.0, 2.0, 3.0]


def test_dimension_is_known_without_loading_the_model():
    embedder = FastEmbedEmbedder(MODEL)

    assert embedder.dimension == 384
    assert embedder._model is None


def test_model_failures_become_embedding_errors():
    class Broken:
        def passage_embed(self, texts, batch_size):
            raise RuntimeError("onnx session crashed")

        def query_embed(self, texts):
            raise RuntimeError("onnx session crashed")

    embedder = embedder_with(Broken())

    with pytest.raises(EmbeddingError, match="2 passages failed"):
        embedder.embed_documents(["a", "b"])
    with pytest.raises(EmbeddingError, match="query failed"):
        embedder.embed_query("a")


def test_model_is_loaded_once_from_the_configured_cache(monkeypatch, tmp_path):
    created = []

    class Recorder(StubModel):
        def __init__(self, model_name, cache_dir):
            super().__init__()
            created.append((model_name, cache_dir))

    monkeypatch.setattr(fastembed_embedder, "TextEmbedding", Recorder, raising=True)
    # The supported-model check and the dimension lookup use the real class.
    monkeypatch.setattr(
        Recorder, "list_supported_models", staticmethod(lambda: [{"model": MODEL}]), raising=False
    )
    embedder = FastEmbedEmbedder(MODEL, cache_dir=tmp_path / "models")

    embedder.embed_query("a")
    embedder.embed_query("b")

    assert created == [(MODEL, str(tmp_path / "models"))]


def test_download_failure_is_an_embedding_error(monkeypatch):
    class Unreachable:
        @staticmethod
        def list_supported_models():
            return [{"model": MODEL}]

        def __init__(self, model_name, cache_dir):
            raise OSError("could not reach huggingface.co")

    monkeypatch.setattr(fastembed_embedder, "TextEmbedding", Unreachable)

    with pytest.raises(EmbeddingError, match="huggingface.co"):
        FastEmbedEmbedder(MODEL).embed_query("a")
