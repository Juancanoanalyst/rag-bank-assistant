"""Data shapes passed between pipeline stages."""

from pydantic import BaseModel


class RawPage(BaseModel):
    """An HTML page as stored in data/raw, before any cleaning."""

    url: str
    html: str
    filename: str


class Document(BaseModel):
    """A cleaned page: one line of data/clean/documents.jsonl."""

    url: str
    title: str
    section: str
    text: str


class Chunk(BaseModel):
    """A slice of a Document: the unit that is embedded, stored and retrieved."""

    id: str
    url: str
    title: str
    section: str
    chunk_index: int
    text: str

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedder: the page title gives a short chunk its context."""
        return f"{self.title}\n{self.text}"
