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
