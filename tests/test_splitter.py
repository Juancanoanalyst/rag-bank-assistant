import uuid

import pytest

from rag_assistant.chunking import RecursiveTextSplitter
from rag_assistant.models import Document

SENTENCE = "La cuenta de ahorros no tiene cuota de manejo durante el primer año. "
PARAGRAPH = SENTENCE * 4


def make_document(text: str, url: str = "https://www.banco-ejemplo.com.co/personas/a.html"):
    return Document(url=url, title="Cuenta de ahorros", section="personas", text=text)


def test_short_text_is_a_single_chunk():
    assert RecursiveTextSplitter(200, 20).split_text("Texto corto.") == ["Texto corto."]


def test_empty_text_produces_no_chunks():
    assert RecursiveTextSplitter(200, 20).split_text("") == []
    assert RecursiveTextSplitter(200, 20).split_text("  \n\n  ") == []


@pytest.mark.parametrize("size, overlap", [(100, 0), (100, 30), (300, 50), (450, 70)])
def test_no_chunk_exceeds_chunk_size(size, overlap):
    text = "\n\n".join([PARAGRAPH] * 6)

    chunks = RecursiveTextSplitter(size, overlap).split_text(text)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= size for chunk in chunks)


def test_no_content_is_lost():
    words = [f"palabra{number}" for number in range(400)]
    text = " ".join(words)

    chunks = RecursiveTextSplitter(120, 30).split_text(text)

    assert {word for chunk in chunks for word in chunk.split()} == set(words)


def test_prefers_paragraph_boundaries():
    first, second = "Primer párrafo. " * 5, "Segundo párrafo. " * 5
    text = f"{first.strip()}\n\n{second.strip()}"

    chunks = RecursiveTextSplitter(100, 0).split_text(text)

    assert chunks == [first.strip(), second.strip()]


def test_falls_back_to_sentences_before_breaking_words():
    chunks = RecursiveTextSplitter(150, 0).split_text(SENTENCE * 6)

    assert all(chunk.endswith(".") for chunk in chunks)


def test_consecutive_chunks_overlap():
    chunks = RecursiveTextSplitter(150, 80).split_text(SENTENCE * 6)

    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.startswith(SENTENCE.strip())
        assert previous.endswith(SENTENCE.strip())
    assert len(chunks) > len(RecursiveTextSplitter(150, 0).split_text(SENTENCE * 6))


def test_text_without_separators_is_cut_by_length():
    chunks = RecursiveTextSplitter(50, 10).split_text("x" * 120)

    assert [len(chunk) for chunk in chunks] == [40, 50, 50]
    assert "".join([chunks[0]] + [chunk[10:] for chunk in chunks[1:]]) == "x" * 120


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        RecursiveTextSplitter(100, 100)


def test_split_documents_carries_metadata_and_indexes():
    document = make_document("\n\n".join([PARAGRAPH] * 3))

    chunks = list(RecursiveTextSplitter(200, 20).split_documents([document]))

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {(chunk.url, chunk.title, chunk.section) for chunk in chunks} == {
        (document.url, "Cuenta de ahorros", "personas")
    }
    assert chunks[0].embedding_text.startswith("Cuenta de ahorros\n")


def test_chunk_ids_are_valid_unique_and_deterministic():
    documents = [
        make_document(PARAGRAPH * 3, url="https://www.banco-ejemplo.com.co/a.html"),
        make_document(PARAGRAPH * 3, url="https://www.banco-ejemplo.com.co/b.html"),
    ]
    splitter = RecursiveTextSplitter(200, 20)

    first_run = [chunk.id for chunk in splitter.split_documents(documents)]
    second_run = [chunk.id for chunk in splitter.split_documents(documents)]

    assert first_run == second_run
    assert len(set(first_run)) == len(first_run)
    assert all(uuid.UUID(chunk_id) for chunk_id in first_run)


def test_overlap_repeats_final_words_when_sentences_are_longer_than_the_overlap():
    sentences = [
        f"La frase número {number} describe con bastante detalle un producto del banco "
        f"y sus condiciones para clientes nuevos."
        for number in range(30)
    ]

    chunks = RecursiveTextSplitter(450, 70).split_text(" ".join(sentences))

    for previous, current in zip(chunks, chunks[1:], strict=False):
        repeated = next(
            current[:length] for length in range(70, 0, -1) if previous.endswith(current[:length])
        )
        assert 20 <= len(repeated) <= 70
        assert not repeated[0].isspace() and previous[-len(repeated) - 1].isspace()
