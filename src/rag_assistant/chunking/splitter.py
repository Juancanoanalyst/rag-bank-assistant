"""Recursive character splitter.

Text is cut at the most meaningful boundary available (paragraph, then line,
then sentence, then word) so chunks stay under CHUNK_SIZE without breaking
sentences unless there is no other way.
"""

import re
import uuid
from collections.abc import Iterable, Iterator

from rag_assistant.models import Chunk, Document

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")
_WHITESPACE = re.compile(r"\s")


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
        text = text.replace("\r\n", "\n")
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
                break
            if separator in text:
                parts = text.split(separator)
                parts = [part + separator for part in parts[:-1]] + [parts[-1]]
                remaining = separators[position + 1 :]
                return [piece for part in parts if part for piece in self._pieces(part, remaining)]
        # No separator left: cut by length, leaving room for the overlap to be repeated.
        step = self._size - self._overlap
        return [text[start : start + step] for start in range(0, len(text), step)]

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
        """Text to repeat at the start of the next chunk.

        At most chunk_overlap characters, and no more than `room` (what the next
        piece leaves free). Whole trailing pieces are preferred; when even the
        last piece is too long, which is the usual case with sentences, its
        final words are repeated instead.
        """
        budget = min(self._overlap, room)
        tail: list[str] = []
        length = 0
        for piece in reversed(pieces):
            if not piece.strip() or length + len(piece) > budget:
                break
            tail.insert(0, piece)
            length += len(piece)
        if tail or budget <= 0:
            return tail, length

        last = pieces[-1]
        fragment = last[-budget:]
        if _WHITESPACE.search(last):
            # Ordinary prose: start the repeated text on a word boundary, or
            # repeat nothing if not even one whole word fits. Only text with no
            # whitespace at all (cut by length) is repeated from mid-token.
            boundary = _WHITESPACE.search(fragment)
            fragment = fragment[boundary.end() :] if boundary else ""
        return ([fragment], len(fragment)) if fragment.strip() else ([], 0)
