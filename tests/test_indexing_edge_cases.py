import pytest
from fakes import FakeEmbedder
from qdrant_client import QdrantClient

from rag_assistant import factory
from rag_assistant.chunking import RecursiveTextSplitter
from rag_assistant.exceptions import ConfigurationError, IndexingError
from rag_assistant.indexing import Indexer
from rag_assistant.indexing import __main__ as cli
from rag_assistant.models import Chunk, Document
from rag_assistant.scraping.storage import CleanStore
from rag_assistant.vectorstore import QdrantVectorStore

SITE = "https://www.banco-ejemplo.com.co"
COLLECTION = "test_collection"
SIZES_AND_OVERLAPS = [(40, 0), (60, 15), (120, 119), (200, 50), (450, 70)]
NEARLY_FULL_OVERLAP = (40, 39)
LINE_ENDINGS = ["\n", "\r\n"]


class FailingEmbedder(FakeEmbedder):
    """Embeds normally until the given call, which fails like a broken model would."""

    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self._fail_on_call = fail_on_call

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(self.calls) + 1 == self._fail_on_call:
            raise IndexingError("Embedding failed")
        return super().embed_documents(texts)


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


def make_chunk():
    return Chunk(
        id="00000000-0000-0000-0000-000000000001",
        url=f"{SITE}/personas/cuentas.html",
        title="Cuenta de ahorros",
        section="personas",
        chunk_index=0,
        text="Sin cuota de manejo.",
    )


def indexed_urls(client):
    points, _ = client.scroll(COLLECTION, limit=1000, with_payload=True)
    return {point.payload["url"] for point in points}


def mixed_text(line_ending: str) -> str:
    """Unique words in sentences of uneven length, grouped into lines and paragraphs."""
    words = [f"palabra{number}" for number in range(300)]
    sentences = []
    while words:
        length = (3, 7, 12, 5)[len(sentences) % 4]
        sentences.append(" ".join(words[:length]) + ".")
        words = words[length:]
    lines = [" ".join(sentences[start : start + 3]) for start in range(0, len(sentences), 3)]
    paragraphs = [line_ending.join(lines[start : start + 2]) for start in range(0, len(lines), 2)]
    return (line_ending * 2).join(paragraphs)


def word_spans(chunks: list[str], words: list[str]) -> list[tuple[int, int]]:
    """Where each chunk starts and ends, as positions in the original list of unique words."""
    position = {word: index for index, word in enumerate(words)}
    return [(position[chunk.split()[0]], position[chunk.split()[-1]] + 1) for chunk in chunks]


# --- splitter ----------------------------------------------------------------


@pytest.mark.parametrize("line_ending", LINE_ENDINGS)
@pytest.mark.parametrize("size, overlap", [*SIZES_AND_OVERLAPS, NEARLY_FULL_OVERLAP])
def test_chunks_fit_the_size_and_cover_every_word_in_document_order(size, overlap, line_ending):
    text = mixed_text(line_ending)
    words = text.split()

    chunks = RecursiveTextSplitter(size, overlap).split_text(text)

    spans = word_spans(chunks, words)
    assert all(len(chunk) <= size for chunk in chunks)
    assert [chunk.split() for chunk in chunks] == [words[start:end] for start, end in spans]
    assert spans[0][0] == 0
    assert spans[-1][1] == len(words)
    for (previous_start, previous_end), (start, _) in zip(spans, spans[1:], strict=False):
        assert previous_start < start <= previous_end


@pytest.mark.parametrize("line_ending", LINE_ENDINGS)
@pytest.mark.parametrize("size, overlap", SIZES_AND_OVERLAPS)
def test_repeated_text_never_exceeds_chunk_overlap(size, overlap, line_ending):
    text = mixed_text(line_ending)
    words = text.split()

    chunks = RecursiveTextSplitter(size, overlap).split_text(text)

    spans = word_spans(chunks, words)
    for (_, previous_end), (start, _) in zip(spans, spans[1:], strict=False):
        assert len(" ".join(words[start:previous_end])) <= overlap


def test_word_longer_than_chunk_size_is_cut_and_the_text_after_it_is_kept():
    tail = "seguido de un texto normal que conserva todas sus palabras"

    chunks = RecursiveTextSplitter(50, 0).split_text("z" * 120 + " " + tail)

    assert all(len(chunk) <= 50 for chunk in chunks)
    assert "".join(chunks).count("z") == 120
    assert " ".join(chunks).split()[-len(tail.split()) :] == tail.split()


def test_accented_spanish_questions_are_kept_whole_and_measured_in_characters():
    text = (
        "¿Cuál es la tasa de interés? La tasa es del 12 % efectivo anual. "
        "¿Cómo solicito el crédito? Puede pedirlo en línea o en una oficina. "
        "¿Qué pasa si no pago a tiempo? Se cobran intereses de mora según la ley."
    )

    chunks = RecursiveTextSplitter(80, 0).split_text(text)

    assert len(chunks) == 3
    assert all(chunk.startswith("¿") and len(chunk) <= 80 for chunk in chunks)
    assert " ".join(chunks) == text


def test_windows_line_endings_are_normalised_and_keep_paragraphs_together():
    first = "línea uno del primer párrafo\r\nlínea dos del primer párrafo"
    second = "línea uno del segundo párrafo\r\nlínea dos del segundo párrafo"

    chunks = RecursiveTextSplitter(100, 0).split_text(f"{first}\r\n\r\n{second}")

    assert chunks == [first.replace("\r\n", "\n"), second.replace("\r\n", "\n")]


@pytest.mark.parametrize("sentence_length", [48, 49, 50])
def test_sentence_ending_at_the_size_boundary_is_not_broken(sentence_length):
    sentence = "b" * (sentence_length - 1) + "."

    chunks = RecursiveTextSplitter(50, 0).split_text(f"{sentence} {sentence} Fin.")

    assert chunks == [sentence, sentence, "Fin."]


def test_custom_separators_replace_the_default_ones():
    text = "uno|dos|tres|cuatro|cinco|seis|siete|ocho"

    chunks = RecursiveTextSplitter(20, 0, separators=("|",)).split_text(text)

    assert chunks == ["uno|dos|tres|cuatro|", "cinco|seis|siete|", "ocho"]


# --- chunk -------------------------------------------------------------------


def test_embedding_text_is_the_title_then_the_text_and_is_not_a_stored_field():
    chunk = make_chunk()

    assert chunk.embedding_text == "Cuenta de ahorros\nSin cuota de manejo."
    assert "embedding_text" not in chunk.model_dump()


# --- indexer -----------------------------------------------------------------


def test_failure_on_the_first_batch_leaves_an_empty_collection_that_is_not_skipped(
    store, documents
):
    with pytest.raises(IndexingError, match="Embedding failed"):
        make_indexer(store, FailingEmbedder(fail_on_call=1)).index(documents)

    report = make_indexer(store).index(documents, skip_if_indexed=True)

    assert not report.skipped
    assert store.count() == report.chunks > 0


def test_skip_if_indexed_does_not_mistake_a_half_built_collection_for_a_complete_one(
    store, documents
):
    expected = len(list(RecursiveTextSplitter(200, 30).split_documents(documents)))
    with pytest.raises(IndexingError, match="Embedding failed"):
        make_indexer(store, FailingEmbedder(fail_on_call=2)).index(documents)

    report = make_indexer(store).index(documents, skip_if_indexed=True)

    assert not report.skipped
    assert store.count() == expected


def test_document_with_empty_text_is_counted_but_stores_no_chunks(store, client, documents):
    empty = Document(url=f"{SITE}/vacia.html", title="Vacía", section="inicio", text="  \n ")

    report = make_indexer(store).index([*documents, empty])

    assert report.documents == 3
    assert report.chunks == store.count()
    assert indexed_urls(client) == {document.url for document in documents}


def test_documents_without_text_fail_and_keep_the_existing_collection(store, documents):
    empty = Document(url=f"{SITE}/vacia.html", title="Vacía", section="inicio", text="")
    indexer = make_indexer(store)
    first = indexer.index(documents)

    with pytest.raises(IndexingError):
        indexer.index([empty])

    assert store.count() == first.chunks


def test_batch_size_larger_than_the_chunk_count_embeds_everything_in_one_call(store, documents):
    embedder = FakeEmbedder()

    report = make_indexer(store, embedder, batch_size=1000).index(documents)

    assert [len(batch) for batch in embedder.calls] == [report.chunks]
    assert store.count() == report.chunks


def test_reindexing_with_another_embedder_dimension_resizes_the_collection(
    store, client, documents
):
    make_indexer(store, FakeEmbedder(dimension=16)).index(documents)

    report = make_indexer(store, FakeEmbedder(dimension=8)).index(documents)

    assert client.get_collection(COLLECTION).config.params.vectors.size == 8
    assert store.count() == report.chunks


# --- vector store ------------------------------------------------------------


def test_recreate_with_another_dimension_drops_the_points_and_rejects_old_vectors(store, client):
    chunk = make_chunk()
    store.recreate(dimension=16)
    store.upsert([chunk], [[1.0] * 16])

    store.recreate(dimension=8)

    assert store.count() == 0
    assert client.get_collection(COLLECTION).config.params.vectors.size == 8
    with pytest.raises(IndexingError, match="Could not write 1 chunks"):
        store.upsert([chunk], [[1.0] * 16])


# --- factory -----------------------------------------------------------------


def test_unknown_embedder_provider_is_a_configuration_error_naming_the_valid_ones(settings):
    custom = settings.model_copy(update={"embedder_provider": "openai"})

    with pytest.raises(ConfigurationError, match=r"EMBEDDER_PROVIDER='openai'.*fastembed"):
        factory.build_embedder(custom)


# --- command line ------------------------------------------------------------


@pytest.mark.parametrize("argv, documents_left", [(["--skip-if-indexed"], 2), ([], 1)])
def test_cli_rebuilds_a_populated_collection_only_without_skip_if_indexed(
    argv, documents_left, settings, monkeypatch, store, client, documents
):
    make_indexer(store).index(documents)
    CleanStore(settings.clean_data_dir).write(documents[:1])
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "build_indexer", lambda _: make_indexer(store))

    exit_code = cli.main(argv)

    assert exit_code == 0
    assert len(indexed_urls(client)) == documents_left
