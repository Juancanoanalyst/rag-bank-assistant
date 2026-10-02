import pytest
from fakes import FakeEmbedder
from qdrant_client import QdrantClient

from rag_assistant import factory
from rag_assistant.chunking import RecursiveTextSplitter
from rag_assistant.embeddings.fastembed_embedder import FastEmbedEmbedder
from rag_assistant.exceptions import ConfigurationError, IndexingError
from rag_assistant.indexing import Indexer
from rag_assistant.indexing import __main__ as cli
from rag_assistant.models import Chunk, Document
from rag_assistant.scraping.storage import CleanStore
from rag_assistant.vectorstore import QdrantVectorStore

SITE = "https://www.banco-ejemplo.com.co"
COLLECTION = "test_collection"


@pytest.fixture
def client():
    return QdrantClient(":memory:")


@pytest.fixture
def store(client):
    return QdrantVectorStore(client, COLLECTION)


@pytest.fixture
def documents():
    return [
        Document(
            url=f"{SITE}/personas/productos/cuentas/ahorros.html",
            title="Cuenta de ahorros",
            section="personas/productos",
            text="La cuenta de ahorros no tiene cuota de manejo durante el primer año. " * 8,
        ),
        Document(
            url=f"{SITE}/personas/productos/tarjetas.html",
            title="Tarjetas de crédito",
            section="personas/productos",
            text="La tarjeta Oro incluye asistencia en viajes y salas VIP en aeropuertos. " * 8,
        ),
    ]


def make_indexer(store, embedder=None, batch_size=4):
    return Indexer(RecursiveTextSplitter(200, 30), embedder or FakeEmbedder(), store, batch_size)


def payloads(client):
    points, _ = client.scroll(COLLECTION, limit=1000, with_payload=True)
    return [point.payload for point in points]


# --- indexer -----------------------------------------------------------------


def test_index_stores_every_chunk_with_its_metadata(store, client, documents):
    report = make_indexer(store).index(documents)

    stored = payloads(client)
    assert report.documents == 2
    assert report.chunks == store.count() == len(stored) > 2
    assert {payload["url"] for payload in stored} == {document.url for document in documents}
    assert set(stored[0]) == {"url", "title", "section", "chunk_index", "text"}


def test_index_embeds_title_with_text_in_batches(store, documents):
    embedder = FakeEmbedder()

    report = make_indexer(store, embedder, batch_size=4).index(documents)

    assert all(len(batch) <= 4 for batch in embedder.calls)
    assert sum(len(batch) for batch in embedder.calls) == report.chunks
    assert embedder.calls[0][0].startswith("Cuenta de ahorros\n")


def test_reindexing_replaces_previous_content(store, client, documents):
    indexer = make_indexer(store)
    indexer.index(documents)

    report = indexer.index(documents[:1])

    assert store.count() == report.chunks
    assert {payload["url"] for payload in payloads(client)} == {documents[0].url}


def test_skip_if_indexed_leaves_a_populated_collection_alone(store, documents):
    indexer = make_indexer(store)
    first = indexer.index(documents)
    embedder = FakeEmbedder()

    report = make_indexer(store, embedder).index(documents[:1], skip_if_indexed=True)

    assert report.skipped
    assert embedder.calls == []
    assert store.count() == first.chunks


def test_skip_if_indexed_still_indexes_an_empty_store(store, documents):
    report = make_indexer(store).index(documents, skip_if_indexed=True)

    assert not report.skipped
    assert store.count() == report.chunks


def test_index_without_documents_fails_and_keeps_existing_collection(store, documents):
    indexer = make_indexer(store)
    first = indexer.index(documents)

    with pytest.raises(IndexingError, match="no documents"):
        indexer.index([])

    assert store.count() == first.chunks


# --- vector store ------------------------------------------------------------


def test_count_is_zero_before_the_collection_exists(store):
    assert store.count() == 0


def test_upsert_rejects_mismatched_chunks_and_vectors(store):
    store.recreate(dimension=16)
    chunk = Chunk(
        id="00000000-0000-0000-0000-000000000001",
        url=SITE,
        title="t",
        section="inicio",
        chunk_index=0,
        text="texto",
    )

    with pytest.raises(IndexingError, match="1 chunks but 0 vectors"):
        store.upsert([chunk], [])


def test_vector_store_failures_become_indexing_errors(documents):
    unreachable = QdrantClient(url="http://127.0.0.1:1", timeout=1)
    store = QdrantVectorStore(unreachable, COLLECTION)

    with pytest.raises(IndexingError, match="vector store"):
        store.count()
    with pytest.raises(IndexingError, match="Could not create collection"):
        make_indexer(store).index(documents)


# --- factory -----------------------------------------------------------------


def test_factory_builds_components_from_settings(settings):
    custom = settings.model_copy(update={"chunk_size": 300, "chunk_overlap": 40})

    splitter = factory.build_splitter(custom)
    embedder = factory.build_embedder(custom)

    assert len(splitter.split_text("palabra " * 200)[0]) <= 300
    assert isinstance(embedder, FastEmbedEmbedder)
    assert embedder.dimension == 384


def test_unsupported_embedding_model_is_a_configuration_error(settings):
    custom = settings.model_copy(update={"embedding_model": "modelo/que-no-existe"})

    with pytest.raises(ConfigurationError, match="EMBEDDING_MODEL"):
        factory.build_embedder(custom)


# --- command line ------------------------------------------------------------


def test_cli_indexes_the_clean_documents(settings, monkeypatch, store, documents):
    CleanStore(settings.clean_data_dir).write(documents)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "build_indexer", lambda _: make_indexer(store))

    assert cli.main([]) == 0
    assert store.count() > 0


def test_cli_fails_cleanly_without_clean_documents(settings, monkeypatch, store):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "build_indexer", lambda _: make_indexer(store))

    assert cli.main([]) == 1
