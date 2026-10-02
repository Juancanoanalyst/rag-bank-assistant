import shutil

import pytest
import requests
import responses

from rag_assistant.exceptions import ScrapingError
from rag_assistant.scraping import __main__ as cli
from rag_assistant.scraping.crawler import SiteCrawler
from rag_assistant.scraping.storage import CleanStore, RawStore

SITE = "https://www.banco-ejemplo.com.co"
AHORROS = f"{SITE}/personas/productos/cuentas/ahorros.html"
TARJETAS = f"{SITE}/personas/productos/tarjetas.html"
HIPOTECARIO = f"{SITE}/personas/productos/hipotecario/preguntas-frecuentes.html"
PRIVADA = f"{SITE}/zona-privada/resumen.html"
HTML = {"Content-Type": "text/html; charset=utf-8"}


@pytest.fixture
def site(read_fixture):
    """A fake bank site: robots.txt, sitemap and three HTML pages."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        mock.get(f"{SITE}/robots.txt", body=read_fixture("robots.txt"))
        mock.get(f"{SITE}/sitemap.xml", body=read_fixture("sitemap.xml"))
        mock.get(AHORROS, body=read_fixture("html/cuenta-ahorros.html"), headers=HTML)
        mock.get(TARJETAS, body=read_fixture("html/guardada-desde-navegador.html"), headers=HTML)
        mock.get(HIPOTECARIO, body=read_fixture("html/con-canonical.html"), headers=HTML)
        yield mock


def make_crawler(settings, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return SiteCrawler(settings, RawStore(settings.raw_data_dir), sleep=sleeps.append)


def requested_urls(mock) -> list[str]:
    return [call.request.url for call in mock.calls]


def test_crawl_saves_allowed_html_pages(site, settings):
    report = make_crawler(settings).crawl()

    saved = {page.url for page in RawStore(settings.raw_data_dir).pages()}
    assert saved == {AHORROS, TARJETAS, HIPOTECARIO}
    assert (report.saved, report.failed, report.disallowed) == (3, 0, 1)


def test_crawl_never_requests_disallowed_pdf_or_external_urls(site, settings):
    make_crawler(settings).crawl()

    requested = requested_urls(site)
    assert PRIVADA not in requested
    assert not any(url.endswith(".pdf") or "otro-dominio.com" in url for url in requested)


def test_crawl_sends_configured_user_agent(site, settings):
    make_crawler(settings).crawl()

    assert {call.request.headers["User-Agent"] for call in site.calls} == {"test-agent"}


def test_crawl_waits_between_pages_using_robots_crawl_delay(site, settings):
    sleeps = []

    make_crawler(settings, sleeps).crawl()

    assert sleeps == [2.0, 2.0, 2.0]  # robots.txt asks for 2s, settings for 0s


def test_crawl_respects_max_pages(site, settings):
    limited = settings.model_copy(update={"scraper_max_pages": 1})

    report = make_crawler(limited).crawl()

    assert report.saved == 1
    assert len(list(RawStore(limited.raw_data_dir).pages())) == 1


def test_crawl_continues_after_a_failing_page(site, settings):
    site.replace(responses.GET, TARJETAS, status=500)
    site.replace(responses.GET, HIPOTECARIO, body=requests.Timeout("lento"))

    report = make_crawler(settings).crawl()

    assert (report.saved, report.failed) == (1, 2)


def test_crawl_skips_non_html_responses(site, settings):
    site.replace(responses.GET, AHORROS, body="{}", headers={"Content-Type": "application/json"})

    report = make_crawler(settings).crawl()

    assert (report.saved, report.failed) == (2, 1)


def test_crawl_stops_when_site_blocks_robots(settings):
    with responses.RequestsMock() as mock:
        mock.get(f"{SITE}/robots.txt", status=403)

        with pytest.raises(ScrapingError, match="Save pages manually"):
            make_crawler(settings).crawl()

        assert requested_urls(mock) == [f"{SITE}/robots.txt"]


def test_crawl_fails_when_sitemap_is_unavailable(site, settings):
    site.replace(responses.GET, f"{SITE}/sitemap.xml", status=404)

    with pytest.raises(ScrapingError, match="404"):
        make_crawler(settings).crawl()


# --- command line ------------------------------------------------------------


def test_cli_run_crawls_and_cleans(site, settings, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr("rag_assistant.scraping.crawler.time.sleep", lambda _: None)

    assert cli.main(["run"]) == 0

    documents = CleanStore(settings.clean_data_dir).read()
    assert {document.url for document in documents} == {AHORROS, TARJETAS, HIPOTECARIO}


def test_cli_run_falls_back_to_manually_saved_pages(settings, monkeypatch, fixtures_dir):
    """The real target site answers 403 to non-browser clients: `run` must still work."""
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    settings.raw_data_dir.mkdir(parents=True)
    shutil.copy(fixtures_dir / "html" / "guardada-desde-navegador.html", settings.raw_data_dir)

    with responses.RequestsMock() as mock:
        mock.get(f"{SITE}/robots.txt", status=403)
        exit_code = cli.main(["run"])

    assert exit_code == 0
    assert [document.url for document in CleanStore(settings.clean_data_dir).read()] == [TARJETAS]


def test_cli_crawl_reports_failure_with_exit_code(settings, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    with responses.RequestsMock() as mock:
        mock.get(f"{SITE}/robots.txt", status=403)
        assert cli.main(["crawl"]) == 1


def test_cli_clean_with_empty_raw_folder_fails(settings, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    assert cli.main(["clean"]) == 1


def test_crawl_does_not_download_pages_already_stored(site, settings):
    make_crawler(settings).crawl()
    first_run_requests = len(site.calls)
    sleeps = []

    report = make_crawler(settings, sleeps).crawl()

    assert (report.saved, report.cached) == (0, 3)
    assert sleeps == []
    # Second run only re-reads robots.txt and the sitemap.
    assert len(site.calls) - first_run_requests == 2
