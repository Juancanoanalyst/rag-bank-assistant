import shutil

import pytest
import responses
from fakes import FakeEmbedder
from qdrant_client import QdrantClient

from rag_assistant import ingest
from rag_assistant.chunking import RecursiveTextSplitter
from rag_assistant.indexing import Indexer
from rag_assistant.vectorstore import QdrantVectorStore

SITE = "https://www.banco-ejemplo.com.co"


@pytest.fixture
def store():
    return QdrantVectorStore(QdrantClient(":memory:"), "test_collection")


@pytest.fixture
def wired(settings, store, monkeypatch):
    """Point the ingest entrypoint at temporary folders, a fake embedder and in-memory Qdrant."""
    indexer = Indexer(RecursiveTextSplitter(200, 30), FakeEmbedder(), store)
    monkeypatch.setattr(ingest, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest, "build_indexer", lambda _: indexer)
    return settings


def save_page_by_hand(settings, fixtures_dir) -> None:
    settings.raw_data_dir.mkdir(parents=True)
    shutil.copy(fixtures_dir / "html" / "guardada-desde-navegador.html", settings.raw_data_dir)


def test_ingest_crawls_cleans_and_indexes(wired, store, read_fixture):
    html = {"Content-Type": "text/html; charset=utf-8"}
    page = f"{SITE}/personas/productos/cuentas/ahorros.html"
    sitemap = f"<urlset><url><loc>{page}</loc></url></urlset>"
    with responses.RequestsMock() as mock:
        mock.get(f"{SITE}/robots.txt", status=404)
        mock.get(f"{SITE}/sitemap.xml", body=sitemap)
        mock.get(page, body=read_fixture("html/cuenta-ahorros.html"), headers=html)

        exit_code = ingest.main([])

    assert exit_code == 0
    assert store.count() > 0


def test_ingest_does_nothing_when_already_indexed(wired, store, fixtures_dir):
    save_page_by_hand(wired, fixtures_dir)
    with responses.RequestsMock() as mock:
        mock.get(f"{SITE}/robots.txt", status=403)
        ingest.main([])
    indexed = store.count()

    with responses.RequestsMock():  # any HTTP request would fail the test
        exit_code = ingest.main([])

    assert exit_code == 0
    assert store.count() == indexed > 0


def test_ingest_force_rebuilds_even_when_indexed(wired, store, fixtures_dir):
    save_page_by_hand(wired, fixtures_dir)
    with responses.RequestsMock() as mock:
        mock.get(f"{SITE}/robots.txt", status=403)
        ingest.main([])
        ingest.main(["--force"])

        assert len(mock.calls) == 2


def test_ingest_survives_a_blocked_crawl_using_saved_pages(wired, store, fixtures_dir):
    save_page_by_hand(wired, fixtures_dir)
    with responses.RequestsMock() as mock:
        mock.get(f"{SITE}/robots.txt", status=403)

        exit_code = ingest.main([])

    assert exit_code == 0
    assert store.count() > 0


def test_ingest_fails_when_there_is_nothing_to_index(wired, store):
    with responses.RequestsMock() as mock:
        mock.get(f"{SITE}/robots.txt", status=403)

        exit_code = ingest.main([])

    assert exit_code == 1
    assert store.count() == 0
