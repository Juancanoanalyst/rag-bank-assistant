import json
import shutil

import pytest

from rag_assistant.exceptions import ScrapingError
from rag_assistant.models import Document, RawPage
from rag_assistant.scraping.cleaner import HtmlCleaner
from rag_assistant.scraping.storage import CleanStore, RawStore, infer_url
from rag_assistant.scraping.urls import section_from_url, spread_by_section

SITE = "https://www.banco-ejemplo.com.co"
AHORROS_URL = f"{SITE}/personas/productos/cuentas/ahorros.html"


def raw_page(read_fixture, name: str, url: str = AHORROS_URL) -> RawPage:
    return RawPage(url=url, html=read_fixture(f"html/{name}"), filename=name)


# --- cleaner ---------------------------------------------------------------


def test_clean_keeps_main_content_and_metadata(read_fixture):
    document = HtmlCleaner(min_text_chars=200).clean(raw_page(read_fixture, "cuenta-ahorros.html"))

    assert document.url == AHORROS_URL
    assert "Cuenta de ahorros" in document.title
    assert document.section == "personas/productos"
    assert "recibir intereses sobre tu saldo" in document.text
    assert "Sin cuota de manejo durante el primer año" in document.text  # list items
    assert "0,50 %" in document.text  # table cells


def test_clean_drops_boilerplate(read_fixture):
    document = HtmlCleaner(min_text_chars=200).clean(raw_page(read_fixture, "cuenta-ahorros.html"))

    for noise in ("cookies", "dataLayer", "Banca en línea", "Política de privacidad"):
        assert noise not in document.text


def test_clean_returns_none_for_pages_without_text(read_fixture):
    assert (
        HtmlCleaner(min_text_chars=200).clean(raw_page(read_fixture, "sin-contenido.html")) is None
    )


def test_clean_survives_garbage_input():
    page = RawPage(url=AHORROS_URL, html="\x00\x01 not html at all", filename="roto.html")

    assert HtmlCleaner(min_text_chars=200).clean(page) is None


def test_clean_all_keeps_one_document_per_url(read_fixture):
    pages = [
        raw_page(read_fixture, "cuenta-ahorros.html"),
        raw_page(read_fixture, "con-canonical.html"),
    ]

    documents = list(HtmlCleaner(min_text_chars=200).clean_all(pages))

    assert len(documents) == 1


@pytest.mark.parametrize(
    "url, expected",
    [
        (f"{SITE}/personas/productos/cuentas/ahorros.html", "personas/productos"),
        (f"{SITE}/personas/productos/", "personas/productos"),
        (f"{SITE}/empresas.html", "inicio"),
        (f"{SITE}/", "inicio"),
        ("local://pagina.html", "inicio"),
    ],
)
def test_section_from_url(url, expected):
    assert section_from_url(url) == expected


# --- url inference for pages saved by hand ----------------------------------


def test_infer_url_from_browser_saved_comment(read_fixture):
    html = read_fixture("html/guardada-desde-navegador.html")

    assert infer_url(html) == f"{SITE}/personas/productos/tarjetas.html"


def test_infer_url_prefers_canonical_over_og_url(read_fixture):
    html = read_fixture("html/con-canonical.html")

    assert infer_url(html) == f"{SITE}/personas/productos/hipotecario/preguntas-frecuentes.html"


def test_infer_url_returns_none_without_hints(read_fixture):
    assert infer_url(read_fixture("html/cuenta-ahorros.html")) is None


# --- raw store ---------------------------------------------------------------


def test_raw_store_round_trip(tmp_path):
    store = RawStore(tmp_path / "raw")

    path = store.save(AHORROS_URL, "<html><body>hola</body></html>")
    pages = list(store.pages())

    assert path.exists() and path.suffix == ".html"
    assert [(page.url, page.html) for page in pages] == [
        (AHORROS_URL, "<html><body>hola</body></html>")
    ]


def test_raw_store_saving_same_url_twice_overwrites(tmp_path):
    store = RawStore(tmp_path / "raw")

    store.save(AHORROS_URL, "version 1")
    store.save(AHORROS_URL, "version 2")

    assert [page.html for page in store.pages()] == ["version 2"]


def test_raw_store_reads_manually_saved_pages(tmp_path, fixtures_dir):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in ("guardada-desde-navegador.html", "con-canonical.html", "cuenta-ahorros.html"):
        shutil.copy(fixtures_dir / "html" / name, raw_dir / name)

    urls = {page.filename: page.url for page in RawStore(raw_dir).pages()}

    assert urls["guardada-desde-navegador.html"] == f"{SITE}/personas/productos/tarjetas.html"
    assert urls["con-canonical.html"].endswith("/hipotecario/preguntas-frecuentes.html")
    assert urls["cuenta-ahorros.html"] == "local://cuenta-ahorros.html"


def test_raw_store_missing_directory_yields_nothing(tmp_path):
    assert list(RawStore(tmp_path / "no-existe").pages()) == []


# --- clean store -------------------------------------------------------------


def test_clean_store_writes_jsonl_with_expected_fields(tmp_path):
    store = CleanStore(tmp_path / "clean")
    document = Document(url=AHORROS_URL, title="Cuenta de ahorros", section="personas", text="Año")

    assert store.write([document]) == 1

    line = store.path.read_text(encoding="utf-8").strip()
    assert json.loads(line) == {
        "url": AHORROS_URL,
        "title": "Cuenta de ahorros",
        "section": "personas",
        "text": "Año",
    }
    assert "Año" in line  # stored as UTF-8, not \u escapes
    assert store.read() == [document]


def test_clean_store_write_replaces_previous_content(tmp_path):
    store = CleanStore(tmp_path / "clean")
    first = Document(url="https://a.co/1", title="1", section="inicio", text="uno")
    second = Document(url="https://a.co/2", title="2", section="inicio", text="dos")

    store.write([first])
    store.write([second])

    assert store.read() == [second]


def test_clean_store_read_before_write_raises(tmp_path):
    with pytest.raises(ScrapingError, match="does not exist"):
        CleanStore(tmp_path / "clean").read()


# --- text tidying and page selection -----------------------------------------


def test_clean_removes_cookie_banner_and_empty_list_markers():
    html = """<html><head><title>Bre-B</title></head><body>
      <div class="Cookies__base"><p>Utilizamos cookies propias y de terceros para mejorar
      nuestros servicios y mostrar publicidad relacionada con sus preferencias.</p></div>
      <main><h1>¿Cómo recibir dinero con Bre-B?</h1>
      <p>Con Bre-B recibes transferencias inmediatas desde cualquier banco usando una llave
      como tu número de celular, sin costo y a cualquier hora del día.</p>
      <ul><li> </li><li>Abre la aplicación y entra a la opción Bre-B del menú principal.</li>
      <li>Registra tu llave y confirma con tu clave para empezar a recibir dinero.</li></ul>
      </main></body></html>"""
    page = RawPage(url=f"{SITE}/personas/blog/bre-b.html", html=html, filename="bre-b.html")

    document = HtmlCleaner(min_text_chars=50).clean(page)

    assert "cookies" not in document.text
    assert "Registra tu llave" in document.text
    assert all(line.strip() not in ("", "-") for line in document.text.splitlines())


def test_spread_by_section_takes_turns_between_sections():
    urls = [
        f"{SITE}/personas/productos/a.html",
        f"{SITE}/personas/productos/b.html",
        f"{SITE}/personas/productos/c.html",
        f"{SITE}/personas/blog/x.html",
        f"{SITE}/empresas/productos/y.html",
    ]

    selected = spread_by_section(urls, limit=4)

    assert selected == [
        f"{SITE}/personas/productos/a.html",
        f"{SITE}/personas/blog/x.html",
        f"{SITE}/empresas/productos/y.html",
        f"{SITE}/personas/productos/b.html",
    ]


def test_spread_by_section_returns_everything_when_under_limit():
    urls = [f"{SITE}/personas/blog/x.html", f"{SITE}/personas/blog/y.html"]

    assert spread_by_section(urls, limit=10) == urls


def test_raw_store_urls_lists_only_files_still_on_disk(tmp_path):
    store = RawStore(tmp_path / "raw")
    kept = store.save(AHORROS_URL, "a")
    removed = store.save(f"{SITE}/otra.html", "b")
    removed.unlink()

    assert kept.exists()
    assert store.urls() == {AHORROS_URL}
