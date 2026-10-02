import pytest
import responses

from rag_assistant.models import Document, RawPage
from rag_assistant.scraping import __main__ as cli
from rag_assistant.scraping.cleaner import HtmlCleaner
from rag_assistant.scraping.crawler import SiteCrawler
from rag_assistant.scraping.robots import RobotsPolicy
from rag_assistant.scraping.sitemap import MAX_INDEX_DEPTH, collect_urls, parse_sitemap
from rag_assistant.scraping.storage import MANIFEST_NAME, CleanStore, RawStore, infer_url

SITE = "https://www.banco-ejemplo.com.co"
AHORROS = f"{SITE}/personas/productos/cuentas/ahorros.html"
TARJETAS = f"{SITE}/personas/productos/tarjetas.html"
HIPOTECARIO = f"{SITE}/personas/productos/hipotecario/preguntas-frecuentes.html"
HTML = {"Content-Type": "text/html; charset=utf-8"}
NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
BOM = chr(0xFEFF)

PARAGRAPH = (
    "<p>La cuenta de ahorros te permite recibir intereses sobre tu saldo, retirar en "
    "cajeros de todo el país y hacer transferencias sin costo desde la aplicación.</p>"
)


def urlset(*urls: str) -> str:
    entries = "".join(f"<url><loc>{url}</loc></url>" for url in urls)
    return f'<urlset xmlns="{NAMESPACE}">{entries}</urlset>'


def sitemap_index(*urls: str) -> str:
    entries = "".join(f"<sitemap><loc>{url}</loc></sitemap>" for url in urls)
    return f'<sitemapindex xmlns="{NAMESPACE}">{entries}</sitemapindex>'


@pytest.fixture
def site(read_fixture):
    """A fake bank site whose robots.txt and sitemap each test completes with pages."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as mock:
        mock.get(f"{SITE}/robots.txt", body=read_fixture("robots.txt"))
        mock.get(f"{SITE}/sitemap.xml", body=read_fixture("sitemap.xml"))
        yield mock


def make_crawler(settings, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return SiteCrawler(settings, RawStore(settings.raw_data_dir), sleep=sleeps.append)


# --- sitemap -----------------------------------------------------------------


def test_parse_sitemap_without_namespace():
    xml = f"<urlset><url><loc>{AHORROS}</loc></url></urlset>"

    assert parse_sitemap(xml) == ([AHORROS], [])


@pytest.mark.parametrize("prefix", [BOM, "\n\n  "], ids=["bom", "whitespace"])
def test_parse_sitemap_tolerates_bom_or_leading_whitespace(prefix):
    xml = prefix + '<?xml version="1.0" encoding="UTF-8"?>\n' + urlset(AHORROS)

    assert parse_sitemap(xml) == ([AHORROS], [])


def test_parse_sitemap_ignores_image_locations():
    xml = (
        f'<urlset xmlns="{NAMESPACE}" '
        'xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">'
        f"<url><loc>{AHORROS}</loc>"
        f"<image:image><image:loc>{SITE}/content/dam/banner-ahorros</image:loc></image:image>"
        "</url></urlset>"
    )

    pages, _ = parse_sitemap(xml)

    assert pages == [AHORROS]


def test_collect_urls_stops_following_indexes_beyond_max_depth():
    requested = []

    def fetch(url: str) -> str:
        requested.append(url)
        level = int(url.rsplit("-", 1)[-1].removesuffix(".xml"))
        return sitemap_index(f"{SITE}/nivel-{level + 1}.xml")

    urls = collect_urls(f"{SITE}/nivel-0.xml", fetch, limit=10)

    assert urls == []
    assert requested == [f"{SITE}/nivel-{level}.xml" for level in range(MAX_INDEX_DEPTH + 1)]


def test_collect_urls_survives_a_sitemap_cycle():
    documents = {
        f"{SITE}/a.xml": sitemap_index(f"{SITE}/b.xml", f"{SITE}/paginas.xml"),
        f"{SITE}/b.xml": sitemap_index(f"{SITE}/a.xml", f"{SITE}/b.xml"),
        f"{SITE}/paginas.xml": urlset(AHORROS),
    }
    requested = []

    def fetch(url: str) -> str:
        requested.append(url)
        return documents[url]

    urls = collect_urls(f"{SITE}/a.xml", fetch, limit=10)

    assert urls == [AHORROS]
    assert sorted(requested) == sorted(documents)


# --- robots ------------------------------------------------------------------


def test_robots_group_for_this_user_agent_overrides_wildcard_group():
    rules = (
        "User-agent: *\nDisallow: /personas/\nCrawl-delay: 1\n\n"
        "User-agent: test-agent\nDisallow: /empresas/\nCrawl-delay: 7\n"
    )

    policy = RobotsPolicy.from_text(rules, "test-agent")

    assert policy.allows(f"{SITE}/personas/productos.html")
    assert not policy.allows(f"{SITE}/empresas/productos.html")
    assert policy.crawl_delay == 7.0


# --- url inference -----------------------------------------------------------


@pytest.mark.parametrize(
    "tag",
    [
        f'<link href="{AHORROS}" rel="canonical">',
        f"<link rel='canonical' href='{AHORROS}'>",
        f'<LINK REL="canonical" HREF="{AHORROS}" />',
        f'<meta content="{AHORROS}" property="og:url">',
    ],
)
def test_infer_url_accepts_attribute_order_quotes_and_case_variants(tag):
    assert infer_url(f"<html><head>{tag}</head><body></body></html>") == AHORROS


def test_infer_url_skips_relative_canonical_and_uses_next_hint():
    html = (
        f"<!-- saved from url=(0070){AHORROS} -->"
        '<html><head><link rel="canonical" href="/personas/productos/cuentas/ahorros.html">'
        "</head><body></body></html>"
    )

    assert infer_url(html) == AHORROS


def test_infer_url_decodes_html_entities_in_canonical():
    html = f'<head><link rel="canonical" href="{SITE}/buscar?tipo=cuenta&amp;pagina=2"></head>'

    assert infer_url(html) == f"{SITE}/buscar?tipo=cuenta&pagina=2"


# --- raw store ---------------------------------------------------------------


def test_raw_store_reads_htm_files_and_ignores_other_extensions(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "tarjetas.htm").write_text("<html>tarjetas</html>", encoding="utf-8")
    (raw_dir / "notas.txt").write_text("no es una página", encoding="utf-8")

    pages = list(RawStore(raw_dir).pages())

    assert [(page.filename, page.url) for page in pages] == [
        ("tarjetas.htm", "local://tarjetas.htm")
    ]


def test_raw_store_reads_non_utf8_file_without_failing(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    html = f"<!-- saved from url=(0070){AHORROS} --><html><body>Año de ahorro</body></html>"
    (raw_dir / "latin1.html").write_bytes(html.encode("latin-1"))

    pages = list(RawStore(raw_dir).pages())

    assert [page.url for page in pages] == [AHORROS]
    assert "o de ahorro" in pages[0].html


def test_raw_store_keeps_urls_that_differ_only_in_query_string(tmp_path):
    store = RawStore(tmp_path / "raw")
    first, second = f"{SITE}/buscar.html?pagina=1", f"{SITE}/buscar.html?pagina=2"

    first_path = store.save(first, "uno")
    second_path = store.save(second, "dos")

    assert first_path != second_path
    assert {page.url: page.html for page in store.pages()} == {first: "uno", second: "dos"}
    assert store.urls() == {first, second}


def test_raw_store_ignores_blank_manifest_lines(tmp_path):
    store = RawStore(tmp_path / "raw")
    store.save(AHORROS, "a")
    with (tmp_path / "raw" / MANIFEST_NAME).open("a", encoding="utf-8") as manifest:
        manifest.write("\n   \n")
    store.save(TARJETAS, "b")

    assert store.urls() == {AHORROS, TARJETAS}


def test_raw_store_survives_manifest_line_truncated_by_an_interrupted_crawl(tmp_path):
    store = RawStore(tmp_path / "raw")
    store.save(AHORROS, "a")
    with (tmp_path / "raw" / MANIFEST_NAME).open("a", encoding="utf-8") as manifest:
        manifest.write('{"url": "https://www.banco-ejemplo.com.co/personas/tar')

    assert store.urls() == {AHORROS}
    assert [page.url for page in store.pages()] == [AHORROS]


# --- clean store and cleaner -------------------------------------------------


def test_clean_store_write_with_no_documents_keeps_the_previous_file(tmp_path):
    store = CleanStore(tmp_path / "clean")
    previous = Document(url=AHORROS, title="Ahorros", section="personas", text="texto")
    store.write([previous])

    assert store.write(iter(())) == 0

    assert store.read() == [previous]
    assert list(store.path.parent.iterdir()) == [store.path]


def test_clean_title_falls_back_to_url_when_page_has_no_title():
    html = f"<html><body><div>{PARAGRAPH}{PARAGRAPH}</div></body></html>"
    page = RawPage(url=AHORROS, html=html, filename="sin-titulo.html")

    document = HtmlCleaner(min_text_chars=50).clean(page)

    assert document.title == AHORROS


# --- crawler -----------------------------------------------------------------


def test_crawl_reports_every_failure_without_raising_or_storing(site, settings):
    for url in (AHORROS, TARJETAS, HIPOTECARIO):
        site.get(url, status=503)

    report = make_crawler(settings).crawl()

    assert (report.saved, report.failed, report.cached) == (0, 3, 0)
    assert list(RawStore(settings.raw_data_dir).pages()) == []


def test_crawl_with_no_usable_sitemap_urls_requests_nothing_else(site, settings):
    unusable = urlset(f"{SITE}/documentos/tarifas.PDF", "https://otro-dominio.com/pagina.html")
    site.replace(responses.GET, f"{SITE}/sitemap.xml", body=unusable)
    sleeps = []

    report = make_crawler(settings, sleeps).crawl()

    assert (report.saved, report.failed, report.disallowed) == (0, 0, 0)
    assert sleeps == []
    assert [call.request.url for call in site.calls] == [
        f"{SITE}/robots.txt",
        f"{SITE}/sitemap.xml",
    ]


def test_crawl_with_max_pages_above_sitemap_size_saves_every_page(site, settings):
    for url in (AHORROS, TARJETAS, HIPOTECARIO):
        site.get(url, body="<html><body>hola</body></html>", headers=HTML)
    generous = settings.model_copy(update={"scraper_max_pages": 500})

    report = make_crawler(generous).crawl()

    assert report.saved == 3
    assert RawStore(generous.raw_data_dir).urls() == {AHORROS, TARJETAS, HIPOTECARIO}


def test_crawl_keeps_accents_when_page_declares_charset_only_in_meta_tag(site, settings):
    site.replace(responses.GET, f"{SITE}/sitemap.xml", body=urlset(AHORROS))
    body = '<html><head><meta charset="utf-8"></head><body>Año de ahorro</body></html>'
    site.get(AHORROS, body=body.encode("utf-8"), headers={"Content-Type": "text/html"})

    make_crawler(settings).crawl()

    pages = list(RawStore(settings.raw_data_dir).pages())
    assert "Año de ahorro" in pages[0].html


# --- command line ------------------------------------------------------------


@pytest.mark.parametrize("argv", [["borrar"], []])
def test_cli_rejects_invalid_or_missing_command(argv, settings, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert "crawl" in capsys.readouterr().err
    assert not settings.raw_data_dir.exists()
