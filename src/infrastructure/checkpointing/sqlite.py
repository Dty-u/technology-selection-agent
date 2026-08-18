"""Lazy, local SQLite checkpoint lifecycle helpers."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config.settings import CHECKPOINT_DB_PATH


@contextmanager
def open_sqlite_checkpointer(
    database_path: str | Path = CHECKPOINT_DB_PATH,
) -> Iterator[SqliteSaver]:
    """Open a strict SQLite saver without connecting during module import."""
    resolved_path = Path(database_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved_path, check_same_thread=False)
    serializer = JsonPlusSerializer(allowed_msgpack_modules=None)
    try:
        yield SqliteSaver(connection, serde=serializer)
    finally:
        connection.close()


def close_session(checkpointer: SqliteSaver, thread_id: str) -> None:
    """Delete checkpoint state for exactly one explicit workflow thread."""
    normalized_thread_id = str(thread_id or "").strip()
    if not normalized_thread_id:
        raise ValueError("thread_id must not be empty")
    checkpointer.delete_thread(normalized_thread_id)
