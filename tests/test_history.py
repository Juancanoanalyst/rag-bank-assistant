import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import pytest

from rag_assistant.exceptions import HistoryError
from rag_assistant.history import SQLiteHistoryRepository
from rag_assistant.models import StoredMessage

START = datetime(2026, 4, 10, 15, 0, tzinfo=UTC)


@pytest.fixture
def repository(tmp_path):
    return SQLiteHistoryRepository(tmp_path / "history" / "history.db")


def message(session_id: str, role: str, content: str, minute: int = 0, **extra) -> StoredMessage:
    return StoredMessage(
        session_id=session_id,
        role=role,
        content=content,
        timestamp=START + timedelta(minutes=minute),
        **extra,
    )


def test_messages_round_trip_with_all_their_fields(repository):
    answer = message(
        "s1",
        "assistant",
        "Un CDT es un depósito a término.",
        minute=1,
        latency_ms=1234.5,
        retrieved_urls=["https://a.co/cdt.html", "https://a.co/inversion.html"],
        rerank_scores=[0.82, 0.41],
        answered=True,
    )
    question = message("s1", "user", "¿Qué es un CDT?")

    repository.add([question, answer])

    assert repository.session("s1") == [question, answer]


def test_table_has_the_required_columns(repository, tmp_path):
    with sqlite3.connect(tmp_path / "history" / "history.db") as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}

    assert {
        "session_id",
        "role",
        "content",
        "timestamp",
        "latency_ms",
        "retrieved_urls",
        "rerank_scores",
    } <= columns


def test_recent_returns_last_n_messages_oldest_first(repository):
    repository.add([message("s1", "user", f"mensaje {number}", number) for number in range(6)])

    recent = repository.recent("s1", limit=3)

    assert [item.content for item in recent] == ["mensaje 3", "mensaje 4", "mensaje 5"]


@pytest.mark.parametrize("limit", [0, -1])
def test_recent_with_no_limit_returns_nothing(repository, limit):
    repository.add([message("s1", "user", "hola")])

    assert repository.recent("s1", limit=limit) == []


def test_sessions_are_isolated_from_each_other(repository):
    repository.add([message("s1", "user", "de la sesión uno"), message("s2", "user", "de la dos")])

    assert [item.content for item in repository.session("s1")] == ["de la sesión uno"]
    assert [item.content for item in repository.recent("s2", 10)] == ["de la dos"]
    assert repository.session("desconocida") == []
    assert len(repository.all()) == 2


def test_history_persists_across_repository_instances(tmp_path):
    path = tmp_path / "history.db"
    SQLiteHistoryRepository(path).add([message("s1", "user", "¿sigues ahí?")])

    assert [item.content for item in SQLiteHistoryRepository(path).session("s1")] == [
        "¿sigues ahí?"
    ]


def test_session_id_is_treated_as_data_not_sql(repository):
    hostile = "s1'; DROP TABLE messages; --"

    repository.add([message(hostile, "user", "hola")])

    assert [item.session_id for item in repository.all()] == [hostile]


def test_add_is_atomic(repository):
    invalid = message("s1", "user", "segundo").model_copy(update={"role": "robot"})

    with pytest.raises(HistoryError):
        repository.add([message("s1", "user", "primero"), invalid])

    assert repository.all() == []


def test_concurrent_writers_do_not_lose_messages(repository):
    def write(worker: int) -> None:
        for number in range(10):
            repository.add([message(f"s{worker}", "user", f"{worker}-{number}")])

    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(repository.all()) == 80
    assert [item.content for item in repository.session("s3")] == [f"3-{n}" for n in range(10)]


def test_unusable_database_path_is_a_history_error(tmp_path):
    blocker = tmp_path / "archivo"
    blocker.write_text("no soy un directorio")

    with pytest.raises(HistoryError, match="Could not open"):
        SQLiteHistoryRepository(blocker / "history.db")
