import pytest
from fakes import FakeEmbedder, FakeReranker
from qdrant_client import QdrantClient

from rag_assistant import factory
from rag_assistant.chunking import RecursiveTextSplitter
from rag_assistant.exceptions import ConfigurationError, RetrievalError
from rag_assistant.indexing import Indexer
from rag_assistant.models import Chunk, Document, RetrievedChunk
from rag_assistant.reranking.fastembed_reranker import FastEmbedReranker, _sigmoid
from rag_assistant.reranking.noop import NoOpReranker
from rag_assistant.retrieval import Retriever
from rag_assistant.vectorstore import QdrantVectorStore

SITE = "https://www.banco-ejemplo.com.co"
RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
TOPICS = {
    "ahorros": "cuenta ahorros intereses saldo retiros cajeros",
    "hipotecario": "crédito hipotecario vivienda plazo financiación cuota",
    "tarjetas": "tarjeta crédito puntos millas viajes salas",
}


@pytest.fixture
def store():
    store = QdrantVectorStore(QdrantClient(":memory:"), "test_collection")
    documents = [
        Document(url=f"{SITE}/{name}.html", title=name, section="personas", text=text)
        for name, text in TOPICS.items()
    ]
    Indexer(RecursiveTextSplitter(200, 0), FakeEmbedder(64), store).index(documents)
    return store


def candidate(text: str, vector_score: float = 0.5) -> RetrievedChunk:
    chunk = Chunk(id=text, url=SITE, title="t", section="personas", chunk_index=0, text=text)
    return RetrievedChunk(chunk=chunk, vector_score=vector_score)


def make_retriever(store, reranker=None, top_k=3, rerank_top_n=2):
    return Retriever(FakeEmbedder(64), store, reranker or FakeReranker(), top_k, rerank_top_n)


# --- vector search -----------------------------------------------------------


def test_search_returns_closest_chunks_first_with_metadata(store):
    results = store.search(FakeEmbedder(64).embed_query("crédito hipotecario vivienda"), top_k=3)

    scores = [item.vector_score for item in results]
    assert results[0].chunk.title == "hipotecario"
    assert results[0].chunk.url == f"{SITE}/hipotecario.html"
    assert results[0].rerank_score is None
    assert scores == sorted(scores, reverse=True)


def test_search_respects_top_k(store):
    assert len(store.search(FakeEmbedder(64).embed_query("cuenta"), top_k=2)) == 2


def test_search_on_missing_collection_explains_what_to_do():
    empty = QdrantVectorStore(QdrantClient(":memory:"), "sin_indexar")

    with pytest.raises(RetrievalError, match="run the indexing step"):
        empty.search([0.0] * 64, top_k=3)


def test_search_failures_become_retrieval_errors():
    unreachable = QdrantVectorStore(QdrantClient(url="http://127.0.0.1:1", timeout=1), "c")

    with pytest.raises(RetrievalError, match="Vector search failed"):
        unreachable.search([0.0] * 64, top_k=3)


# --- retriever ---------------------------------------------------------------


def test_retrieve_reranks_candidates_and_keeps_top_n(store):
    results = make_retriever(store).retrieve("tarjeta crédito viajes")

    assert len(results) == 2
    assert results[0].chunk.title == "tarjetas"
    assert results[0].rerank_score == 1.0
    assert results[0].rerank_score >= results[1].rerank_score


def test_reranker_can_promote_a_candidate_the_vector_search_ranked_lower(store):
    class PreferAhorros(FakeReranker):
        def rerank(self, query, candidates, top_n):
            return super().rerank("ahorros", candidates, top_n)

    by_vector = make_retriever(store, NoOpReranker()).retrieve("crédito hipotecario vivienda")
    reranked = make_retriever(store, PreferAhorros()).retrieve("crédito hipotecario vivienda")

    assert by_vector[0].chunk.title == "hipotecario"
    assert reranked[0].chunk.title == "ahorros"


def test_noop_reranker_keeps_vector_order_and_leaves_no_rerank_score(store):
    results = make_retriever(store, NoOpReranker()).retrieve("cuenta ahorros intereses")

    assert results[0].chunk.title == "ahorros"
    assert len(results) == 2
    assert all(item.rerank_score is None for item in results)


def test_retrieve_on_unindexed_store_raises_retrieval_error():
    empty = QdrantVectorStore(QdrantClient(":memory:"), "sin_indexar")

    with pytest.raises(RetrievalError):
        make_retriever(empty).retrieve("cuenta")


# --- fastembed reranker (no model is loaded in these tests) ------------------


class StubCrossEncoder:
    def __init__(self, logits):
        self._logits = logits

    def rerank(self, query, documents):
        return iter(self._logits[: len(documents)])


def reranker_with(model) -> FastEmbedReranker:
    reranker = FastEmbedReranker(RERANKER_MODEL)
    reranker._model = model
    return reranker


def test_fastembed_reranker_sorts_by_score_and_maps_logits_to_probabilities():
    candidates = [candidate("a"), candidate("b"), candidate("c")]
    reranker = reranker_with(StubCrossEncoder([-2.0, 3.0, 0.0]))

    results = reranker.rerank("pregunta", candidates, top_n=2)

    assert [item.chunk.text for item in results] == ["b", "c"]
    assert results[0].rerank_score == pytest.approx(0.9526, abs=1e-4)
    assert results[1].rerank_score == pytest.approx(0.5)
    assert candidates[0].rerank_score is None


def test_fastembed_reranker_with_no_candidates_does_not_load_the_model():
    reranker = FastEmbedReranker(RERANKER_MODEL)

    assert reranker.rerank("pregunta", [], top_n=5) == []
    assert reranker._model is None


def test_fastembed_reranker_failures_become_retrieval_errors():
    class Broken:
        def rerank(self, query, documents):
            raise RuntimeError("onnx session crashed")

    with pytest.raises(RetrievalError, match="onnx session crashed"):
        reranker_with(Broken()).rerank("pregunta", [candidate("a")], top_n=1)


@pytest.mark.parametrize("logit, expected", [(0.0, 0.5), (1000.0, 1.0), (-1000.0, 0.0)])
def test_sigmoid_is_bounded_for_extreme_logits(logit, expected):
    assert _sigmoid(logit) == pytest.approx(expected, abs=1e-9)


# --- factory -----------------------------------------------------------------


def test_factory_selects_reranker_from_settings(settings):
    disabled = settings.model_copy(update={"reranker_provider": "none"})

    assert isinstance(factory.build_reranker(settings), FastEmbedReranker)
    assert isinstance(factory.build_reranker(disabled), NoOpReranker)


def test_unsupported_reranker_model_is_a_configuration_error(settings):
    custom = settings.model_copy(update={"reranker_model": "modelo/que-no-existe"})

    with pytest.raises(ConfigurationError, match="RERANKER_MODEL"):
        factory.build_reranker(custom)


def test_factory_builds_retriever_with_configured_limits(settings):
    custom = settings.model_copy(update={"top_k": 7, "rerank_top_n": 3})

    retriever = factory.build_retriever(custom)

    assert (retriever._top_k, retriever._rerank_top_n) == (7, 3)


def test_warm_up_runs_both_models_once(store):
    embedder = FakeEmbedder(64)
    calls = []

    class RecordingReranker(FakeReranker):
        def rerank(self, query, candidates, top_n):
            calls.append((query, len(candidates)))
            return super().rerank(query, candidates, top_n)

    Retriever(embedder, store, RecordingReranker(), top_k=3, rerank_top_n=2).warm_up()

    assert calls == [("hola", 1)]
