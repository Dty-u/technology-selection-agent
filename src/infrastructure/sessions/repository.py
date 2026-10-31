"""MySQL history of sessions, scoped as user_id -> project_id -> thread_id."""

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from src.config.settings import MYSQL_MEMORY_URI
from src.infrastructure.memory.mysql import _load_pymysql, parse_mysql_uri


_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class MySQLSessionRepository:
    """Persist conversation history; checkpoints stay workflow-only state."""

    def __init__(
        self,
        database_uri: str = MYSQL_MEMORY_URI,
        *,
        connection_factory: Callable[[], Any] | None = None,
    ) -> None:
        normalized_uri = str(database_uri or "").strip()
        if not normalized_uri:
            raise ValueError("MYSQL_MEMORY_URI is required for session history")
        self._connection_options = parse_mysql_uri(normalized_uri)
        self._connection_factory = connection_factory

    def _connect(self):
        if self._connection_factory is not None:
            return self._connection_factory()
        pymysql = _load_pymysql()
        return pymysql.connect(
            **self._connection_options,
            cursorclass=pymysql.cursors.DictCursor,
        )

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                yield cursor
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def setup_schema(self) -> None:
        """Create history tables; call only from an explicit setup job."""
        statements = [
            statement.strip()
            for statement in _SCHEMA_PATH.read_text(encoding="utf-8").split(";")
            if statement.strip()
        ]
        with self._transaction() as cursor:
            for statement in statements:
                cursor.execute(statement)

    @staticmethod
    def _ensure_project(
        cursor: Any,
        user_id: str,
        project_id: str,
        name: str,
        now: datetime,
    ) -> None:
        cursor.execute(
            "INSERT IGNORE INTO agent_projects "
            "(user_id, project_id, name, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (user_id, project_id, name, now, now),
        )

    def ensure_project(
        self,
        *,
        user_id: str,
        project_id: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Register a project under one user; an existing name is only changed on request."""
        now = _utc_now()
        with self._transaction() as cursor:
            self._ensure_project(cursor, user_id, project_id, name or project_id, now)
            if name:
                cursor.execute(
                    "UPDATE agent_projects SET name = %s, updated_at = %s "
                    "WHERE user_id = %s AND project_id = %s",
                    (name, now, user_id, project_id),
                )
            cursor.execute(
                "SELECT * FROM agent_projects WHERE user_id = %s AND project_id = %s",
                (user_id, project_id),
            )
            return cursor.fetchone()

    def list_projects(self, *, user_id: str) -> list[dict[str, Any]]:
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM agent_projects WHERE user_id = %s "
                "ORDER BY updated_at DESC",
                (user_id,),
            )
            return list(cursor.fetchall())

    def begin_turn(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        request_id: str,
        content: str,
        title: str,
    ) -> None:
        """Record the user's message before the workflow starts."""
        now = _utc_now()
        with self._transaction() as cursor:
            self._ensure_project(cursor, user_id, project_id, project_id, now)
            cursor.execute(
                "UPDATE agent_projects SET updated_at = %s "
                "WHERE user_id = %s AND project_id = %s",
                (now, user_id, project_id),
            )
            cursor.execute(
                "INSERT INTO agent_sessions "
                "(thread_id, user_id, project_id, title, status, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, 'running', %s, %s) "
                "ON DUPLICATE KEY UPDATE status = 'running', updated_at = VALUES(updated_at)",
                (thread_id, user_id, project_id, title, now, now),
            )
            cursor.execute(
                "INSERT IGNORE INTO agent_messages "
                "(thread_id, request_id, role, content, delivery, created_at) "
                "VALUES (%s, %s, 'user', %s, 'pending', %s)",
                (thread_id, request_id, content, now),
            )

    def finish_turn(
        self,
        *,
        thread_id: str,
        request_id: str,
        status: str,
        delivery: str,
        assistant_content: str = "",
        kind: str = "message",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Store the outcome; an unchanged carried-over answer is not a new message."""
        now = _utc_now()
        with self._transaction() as cursor:
            cursor.execute(
                "UPDATE agent_sessions SET status = %s, updated_at = %s "
                "WHERE thread_id = %s",
                (status, now, thread_id),
            )
            cursor.execute(
                "UPDATE agent_messages SET delivery = %s "
                "WHERE thread_id = %s AND request_id = %s AND role = 'user'",
                (delivery, thread_id, request_id),
            )
            if not assistant_content:
                return
            cursor.execute(
                "SELECT content FROM agent_messages "
                "WHERE thread_id = %s AND role = 'assistant' AND kind = %s "
                "ORDER BY id DESC LIMIT 1",
                (thread_id, kind),
            )
            latest = cursor.fetchone()
            if latest and latest["content"] == assistant_content:
                return
            cursor.execute(
                "INSERT IGNORE INTO agent_messages "
                "(thread_id, request_id, role, content, delivery, kind, metadata, created_at) "
                "VALUES (%s, %s, 'assistant', %s, 'sent', %s, %s, %s)",
                (
                    thread_id,
                    request_id,
                    assistant_content,
                    kind,
                    json.dumps(metadata or {}, ensure_ascii=False, default=str),
                    now,
                ),
            )

    def fail_turn(self, *, thread_id: str, request_id: str) -> None:
        """Forget an undelivered turn; the client keeps it locally for retry."""
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM agent_messages "
                "WHERE thread_id = %s AND request_id = %s AND role = 'user'",
                (thread_id, request_id),
            )
            cursor.execute(
                "SELECT COUNT(*) AS message_count FROM agent_messages "
                "WHERE thread_id = %s",
                (thread_id,),
            )
            if cursor.fetchone()["message_count"]:
                cursor.execute(
                    "UPDATE agent_sessions SET status = 'error', updated_at = %s "
                    "WHERE thread_id = %s",
                    (_utc_now(), thread_id),
                )
            else:
                cursor.execute(
                    "DELETE FROM agent_sessions WHERE thread_id = %s",
                    (thread_id,),
                )

    def update_status(self, *, thread_id: str, status: str) -> None:
        """Correct a stale status without making the session look recently active."""
        with self._transaction() as cursor:
            cursor.execute(
                "UPDATE agent_sessions SET status = %s, updated_at = updated_at "
                "WHERE thread_id = %s",
                (status, thread_id),
            )

    def list_sessions(
        self,
        *,
        user_id: str,
        project_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM agent_sessions WHERE user_id = %s AND project_id = %s "
                "ORDER BY updated_at DESC LIMIT %s",
                (user_id, project_id, limit),
            )
            return list(cursor.fetchall())

    def list_user_sessions(self, *, user_id: str, limit: int) -> list[dict[str, Any]]:
        """Return one user's sessions across all of their projects."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM agent_sessions WHERE user_id = %s "
                "ORDER BY updated_at DESC LIMIT %s",
                (user_id, limit),
            )
            return list(cursor.fetchall())

    def get_session(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
    ) -> dict[str, Any] | None:
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM agent_sessions "
                "WHERE thread_id = %s AND user_id = %s AND project_id = %s",
                (thread_id, user_id, project_id),
            )
            return cursor.fetchone()

    def list_messages(self, *, thread_id: str, limit: int) -> list[dict[str, Any]]:
        """Return the latest messages of a session in chronological order."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM agent_messages WHERE thread_id = %s "
                "ORDER BY id DESC LIMIT %s",
                (thread_id, limit),
            )
            return list(reversed(cursor.fetchall()))

    def delete_session(self, *, thread_id: str) -> None:
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM agent_messages WHERE thread_id = %s",
                (thread_id,),
            )
            cursor.execute(
                "DELETE FROM agent_sessions WHERE thread_id = %s",
                (thread_id,),
            )

    def import_session(
        self,
        *,
        user_id: str,
        project_id: str,
        thread_id: str,
        title: str,
        status: str,
        created_at: datetime,
        updated_at: datetime,
        messages: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Copy one browser-stored session once; stored history always wins."""
        with self._transaction() as cursor:
            self._ensure_project(cursor, user_id, project_id, project_id, updated_at)
            cursor.execute(
                "INSERT IGNORE INTO agent_sessions "
                "(thread_id, user_id, project_id, title, status, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (thread_id, user_id, project_id, title, status, created_at, updated_at),
            )
            if cursor.rowcount == 0:
                return False
            for message in messages:
                cursor.execute(
                    "INSERT IGNORE INTO agent_messages "
                    "(thread_id, request_id, role, content, delivery, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (
                        thread_id,
                        message["request_id"],
                        message["role"],
                        message["content"],
                        message["delivery"],
                        message["created_at"],
                    ),
                )
            return True
