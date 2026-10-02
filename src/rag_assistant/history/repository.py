"""Conversation history (Repository pattern).

RAGService and the analytics read and write messages through HistoryRepository
and never see SQL, so the storage engine can change without touching them.
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from rag_assistant.exceptions import HistoryError
from rag_assistant.models import StoredMessage

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT    NOT NULL,
    role           TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content        TEXT    NOT NULL,
    timestamp      TEXT    NOT NULL,
    latency_ms     REAL,
    retrieved_urls TEXT    NOT NULL DEFAULT '[]',
    rerank_scores  TEXT    NOT NULL DEFAULT '[]',
    answered       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);
"""
_COLUMNS = (
    "session_id, role, content, timestamp, latency_ms, retrieved_urls, rerank_scores, answered"
)


class HistoryRepository(ABC):
    @abstractmethod
    def add(self, messages: list[StoredMessage]) -> None:
        """Persist messages atomically, in order."""

    @abstractmethod
    def recent(self, session_id: str, limit: int) -> list[StoredMessage]:
        """The last `limit` messages of a session, oldest first."""

    @abstractmethod
    def session(self, session_id: str) -> list[StoredMessage]:
        """Every message of a session, oldest first."""

    @abstractmethod
    def all(self) -> list[StoredMessage]:
        """Every stored message, oldest first (used by the analytics)."""


class SQLiteHistoryRepository(HistoryRepository):
    """SQLite file on a Docker volume.

    A connection is opened per operation: SQLite connections must not be shared
    between the threads FastAPI uses, and at this volume the cost is negligible.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(_SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise HistoryError(f"Could not open the history database at {path}: {exc}") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:  # commit on success, roll back on error
                yield connection
        finally:
            connection.close()

    def add(self, messages: list[StoredMessage]) -> None:
        rows = [
            (
                message.session_id,
                message.role,
                message.content,
                message.timestamp.isoformat(),
                message.latency_ms,
                json.dumps(message.retrieved_urls),
                json.dumps(message.rerank_scores),
                None if message.answered is None else int(message.answered),
            )
            for message in messages
        ]
        try:
            with self._connect() as connection:
                connection.executemany(
                    f"INSERT INTO messages ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
                )
        except sqlite3.Error as exc:
            raise HistoryError(f"Could not save the conversation: {exc}") from exc

    def recent(self, session_id: str, limit: int) -> list[StoredMessage]:
        if limit <= 0:
            return []
        query = f"SELECT {_COLUMNS} FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?"
        return list(reversed(self._select(query, (session_id, limit))))

    def session(self, session_id: str) -> list[StoredMessage]:
        query = f"SELECT {_COLUMNS} FROM messages WHERE session_id = ? ORDER BY id"
        return self._select(query, (session_id,))

    def all(self) -> list[StoredMessage]:
        return self._select(f"SELECT {_COLUMNS} FROM messages ORDER BY id", ())

    def _select(self, query: str, parameters: tuple) -> list[StoredMessage]:
        try:
            with self._connect() as connection:
                rows = connection.execute(query, parameters).fetchall()
        except sqlite3.Error as exc:
            raise HistoryError(f"Could not read the conversation history: {exc}") from exc
        return [
            StoredMessage(
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                latency_ms=row["latency_ms"],
                retrieved_urls=json.loads(row["retrieved_urls"]),
                rerank_scores=json.loads(row["rerank_scores"]),
                answered=None if row["answered"] is None else bool(row["answered"]),
            )
            for row in rows
        ]
