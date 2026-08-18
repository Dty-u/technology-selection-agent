"""Lazy lifecycle helpers for the MySQL LangGraph checkpointer."""

import re
from contextlib import contextmanager
from typing import Any, Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.config.settings import MYSQL_CHECKPOINT_URI


_JSON_TABLE_CHANNEL_DECLARATION = (
    "channel VARCHAR(150) CHARACTER SET utf8mb4 PATH"
)
_COLLATION_NAME_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _apply_checkpoint_database_collation(
    sql: str,
    collation: str,
) -> str:
    """Align the upstream JSON_TABLE channel with checkpoint table columns."""
    normalized_collation = str(collation or "").strip()
    if not _COLLATION_NAME_PATTERN.fullmatch(normalized_collation):
        raise ValueError("invalid MySQL checkpoint database collation")
    if _JSON_TABLE_CHANNEL_DECLARATION not in sql:
        raise RuntimeError(
            "unsupported langgraph-checkpoint-mysql SELECT template"
        )
    replacement = (
        "channel VARCHAR(150) CHARACTER SET utf8mb4 "
        f"COLLATE {normalized_collation} PATH"
    )
    return sql.replace(_JSON_TABLE_CHANNEL_DECLARATION, replacement)


def _load_pymysql_saver():
    """Import the optional production saver only when a session is opened."""
    try:
        from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
    except ImportError as exc:
        raise RuntimeError(
            "MySQL checkpoint dependency is missing; install "
            "langgraph-checkpoint-mysql[pymysql]==3.0.0"
        ) from exc
    class DatabaseCollationPyMySQLSaver(PyMySQLSaver):
        """Keep package JSON_TABLE expressions compatible with the database."""

        def configure_database_collation(self) -> None:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT @@collation_database")
                row = cursor.fetchone()
            if not row:
                raise RuntimeError("unable to read MySQL database collation")
            collation = (
                row[0]
                if not isinstance(row, dict)
                else next(iter(row.values()))
            )
            normalized = str(collation or "").strip()
            if not _COLLATION_NAME_PATTERN.fullmatch(normalized):
                raise RuntimeError("invalid MySQL database collation")
            self._checkpoint_database_collation = normalized

        def _select_sql(self, where: str) -> str:
            collation = getattr(self, "_checkpoint_database_collation", "")
            if not collation:
                raise RuntimeError(
                    "MySQL checkpoint database collation is not configured"
                )
            return _apply_checkpoint_database_collation(
                super()._select_sql(where),
                collation,
            )

    return DatabaseCollationPyMySQLSaver


@contextmanager
def open_mysql_checkpointer(
    database_uri: str | None = MYSQL_CHECKPOINT_URI,
    *,
    setup: bool = False,
) -> Iterator[Any]:
    """Open a MySQL saver without connecting during module import.

    ``setup`` is deliberately opt-in because it performs schema changes. Normal
    application startup expects the checkpoint schema to have been initialized.
    """
    normalized_uri = str(database_uri or "").strip()
    if not normalized_uri:
        raise ValueError(
            "MYSQL_CHECKPOINT_URI is required for checkpointed sessions"
        )

    saver_type = _load_pymysql_saver()
    with saver_type.from_conn_string(normalized_uri) as checkpointer:
        checkpointer.serde = JsonPlusSerializer(allowed_msgpack_modules=None)
        checkpointer.configure_database_collation()
        if setup:
            checkpointer.setup()
        yield checkpointer


def close_session(checkpointer: Any, thread_id: str) -> None:
    """Delete checkpoint state for exactly one explicit workflow thread."""
    normalized_thread_id = str(thread_id or "").strip()
    if not normalized_thread_id:
        raise ValueError("thread_id must not be empty")
    checkpointer.delete_thread(normalized_thread_id)
