"""Recursive character splitter.

Text is cut at the most meaningful boundary available (paragraph, then line,
then sentence, then word) so chunks stay under CHUNK_SIZE without breaking
sentences unless there is no other way.
"""

import uuid
from collections.abc import Iterable, Iterator

from rag_assistant.models import Chunk, Document

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


class RecursiveTextSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._size = chunk_size
        self._overlap = chunk_overlap
        self._separators = separators

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks of at most chunk_size characters."""
        chunks = self._merge(self._pieces(text, self._separators))
        return [chunk for chunk in chunks if chunk]

    def split_documents(self, documents: Iterable[Document]) -> Iterator[Chunk]:
        """Chunk documents, carrying url, title and section into every chunk."""
        for document in documents:
            for index, text in enumerate(self.split_text(document.text)):
                yield Chunk(
                    # Deterministic id: re-indexing the same page overwrites its chunks.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document.url}#{index}")),
                    url=document.url,
                    title=document.title,
                    section=document.section,
                    chunk_index=index,
                    text=text,
                )

    def _pieces(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Break text into pieces no longer than chunk_size, keeping separators attached."""
        if len(text) <= self._size:
            return [text]
        for position, separator in enumerate(separators):
            if separator == "":
                return [
                    text[start : start + self._size] for start in range(0, len(text), self._size)
                ]
            if separator in text:
                parts = text.split(separator)
                parts = [part + separator for part in parts[:-1]] + [parts[-1]]
                remaining = separators[position + 1 :]
                return [piece for part in parts if part for piece in self._pieces(part, remaining)]
        return [text[start : start + self._size] for start in range(0, len(text), self._size)]

    def _merge(self, pieces: list[str]) -> list[str]:
        """Pack pieces into chunks, repeating the tail of each chunk at the start of the next."""
        chunks: list[str] = []
        current: list[str] = []
        length = 0

        for piece in pieces:
            if current and length + len(piece) > self._size:
                chunks.append("".join(current).strip())
                current, length = self._overlap_tail(current, room=self._size - len(piece))
            current.append(piece)
            length += len(piece)

        if current:
            chunks.append("".join(current).strip())
        return chunks

    def _overlap_tail(self, pieces: list[str], room: int) -> tuple[list[str], int]:
        """Trailing pieces to repeat: at most chunk_overlap characters, and no more than `room`."""
        budget = min(self._overlap, room)
        tail: list[str] = []
        length = 0
        for piece in reversed(pieces):
            if length + len(piece) > budget:
                break
            tail.insert(0, piece)
            length += len(piece)
        return tail, length
