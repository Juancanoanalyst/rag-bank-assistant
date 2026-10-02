import pytest
import requests
import responses

from rag_assistant.exceptions import ScrapingError
from rag_assistant.scraping.robots import RobotsPolicy
from rag_assistant.scraping.sitemap import collect_urls, parse_sitemap

SITE = "https://www.banco-ejemplo.com.co"


def test_parse_urlset_returns_trimmed_page_urls(read_fixture):
    pages, children = parse_sitemap(read_fixture("sitemap.xml"))

    assert children == []
    assert len(pages) == 7
    assert f"{SITE}/personas/productos/tarjetas.html" in pages  # whitespace around <loc> removed


def test_parse_sitemap_index_returns_child_sitemaps(read_fixture):
    pages, children = parse_sitemap(read_fixture("sitemap_index.xml"))

    assert pages == []
    assert children == [f"{SITE}/sitemap-personas.xml", f"{SITE}/sitemap-roto.xml"]


def test_parse_rejects_non_xml():
    with pytest.raises(ScrapingError, match="not valid XML"):
        parse_sitemap("<html><body>Algo salió mal</body></html")


def test_collect_urls_deduplicates_and_keeps_order(read_fixture):
    urls = collect_urls(f"{SITE}/sitemap.xml", lambda _: read_fixture("sitemap.xml"), limit=100)

    assert len(urls) == 6
    assert urls[0] == f"{SITE}/personas/productos/cuentas/ahorros.html"


def test_collect_urls_respects_limit(read_fixture):
    urls = collect_urls(f"{SITE}/sitemap.xml", lambda _: read_fixture("sitemap.xml"), limit=2)

    assert len(urls) == 2


def test_collect_urls_follows_index_and_skips_broken_children(read_fixture):
    def fetch(url: str) -> str:
        if url.endswith("sitemap-roto.xml"):
            raise ScrapingError("HTTP 500")
        name = "sitemap_index.xml" if url.endswith("/sitemap.xml") else "sitemap.xml"
        return read_fixture(name)

    urls = collect_urls(f"{SITE}/sitemap.xml", fetch, limit=100)

    assert len(urls) == 6


def test_collect_urls_fails_when_root_sitemap_fails():
    def fetch(url: str) -> str:
        raise ScrapingError("HTTP 403")

    with pytest.raises(ScrapingError, match="403"):
        collect_urls(f"{SITE}/sitemap.xml", fetch, limit=10)


def test_robots_rules_and_crawl_delay(read_fixture):
    policy = RobotsPolicy.from_text(read_fixture("robots.txt"), "test-agent")

    assert policy.allows(f"{SITE}/personas/productos/tarjetas.html")
    assert not policy.allows(f"{SITE}/zona-privada/resumen.html")
    assert policy.crawl_delay == 2.0


def test_robots_without_crawl_delay_reports_zero():
    assert RobotsPolicy.from_text("User-agent: *\nAllow: /", "test-agent").crawl_delay == 0.0


@responses.activate
@pytest.mark.parametrize("status, allowed", [(404, True), (403, False), (401, False)])
def test_robots_fetch_status_handling(status, allowed):
    responses.get(f"{SITE}/robots.txt", status=status)

    policy = RobotsPolicy.fetch(SITE, "test-agent", requests.Session(), timeout=5)

    assert policy.allows(f"{SITE}/personas.html") is allowed


@responses.activate
def test_robots_fetch_server_error_raises():
    responses.get(f"{SITE}/robots.txt", status=503)

    with pytest.raises(ScrapingError, match="503"):
        RobotsPolicy.fetch(SITE, "test-agent", requests.Session(), timeout=5)


@responses.activate
def test_robots_fetch_network_error_raises():
    responses.get(f"{SITE}/robots.txt", body=requests.ConnectionError("sin red"))

    with pytest.raises(ScrapingError, match="Could not fetch"):
        RobotsPolicy.fetch(SITE, "test-agent", requests.Session(), timeout=5)


def test_robots_honours_wildcard_rules():
    """Real case from the target site, which the stdlib parser silently ignores."""
    rules = "User-agent: *\nAllow: /\nDisallow: *.content.html\nDisallow: /personas/cards"
    policy = RobotsPolicy.from_text(rules, "test-agent")

    assert policy.allows(f"{SITE}/personas/productos.html")
    assert not policy.allows(f"{SITE}/personas/productos.content.html")
    assert not policy.allows(f"{SITE}/personas/cards/oro.html")
